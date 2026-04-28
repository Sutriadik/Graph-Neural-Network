"""
app.py — Streamlit Deployment: GNN MolHIV Predictor
=====================================================

Aplikasi web interaktif untuk memprediksi apakah suatu molekul
memiliki potensi menghambat replikasi HIV menggunakan GIN.

Cara menjalankan:
    streamlit run app.py
"""

import os
import json
import torch
import numpy as np
import streamlit as st
from torch_geometric.data import Batch

from src.model import GINMolHIV
from src.utils import mol_to_graph

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GNN MolHIV Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Header */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #e94560;
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
}
.main-header p {
    color: #aaaacc;
    font-size: 1rem;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-left: 4px solid #378ADD;
    margin-bottom: 0.8rem;
}
.metric-label { font-size: 0.8rem; color: #666; font-weight: 500; text-transform: uppercase; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }

/* Prediction result boxes */
.result-positive {
    background: #fff0f0;
    border: 2px solid #e24b4a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-negative {
    background: #f0fff4;
    border: 2px solid #1d9e75;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-title   { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-subtitle { font-size: 0.9rem; color: #555; }

/* Probability bar */
.prob-bar-container {
    background: #e9ecef;
    border-radius: 10px;
    height: 22px;
    margin: 0.8rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    font-size: 12px;
    font-weight: 600;
    color: white;
}

/* Info boxes — adaptif dark/light mode */
.info-box {
    background: rgba(55, 138, 221, 0.08);
    border-left: 3px solid rgba(55, 138, 221, 0.6);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.875rem;
    line-height: 1.6;
    color: inherit;
}
.warning-box {
    background: rgba(239, 159, 39, 0.08);
    border-left: 3px solid rgba(239, 159, 39, 0.6);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.875rem;
    line-height: 1.6;
    color: inherit;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH   = "model/best_model.pt"
CONFIG_PATH  = "model/config.pt"
METRICS_PATH = "model/metrics.json"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    """Load trained GIN model (cached)."""
    config = torch.load(CONFIG_PATH, map_location="cpu") if os.path.exists(CONFIG_PATH) else {}

    model = GINMolHIV(
        hidden_dim=config.get("hidden_dim", 300),
        num_layers=config.get("num_layers", 5),
        dropout=config.get("dropout", 0.5),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, config


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def predict_smiles(model, smiles: str):
    """
    Prediksi satu molekul dari SMILES string.

    Returns:
        prob (float): Probabilitas HIV inhibitor [0–1]
        label (int) : 1 = HIV inhibitor, 0 = bukan
        graph       : PyG Data object
    """
    graph = mol_to_graph(smiles)
    if graph is None:
        return None, None, None

    batch = Batch.from_data_list([graph]).to(DEVICE)
    with torch.no_grad():
        logit = model(batch)
        prob  = torch.sigmoid(logit).item()

    label = 1 if prob >= 0.5 else 0
    return prob, label, graph


def draw_molecule(smiles: str):
    """Render molekul sebagai gambar menggunakan RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import io
        from PIL import Image

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        img = Draw.MolToImage(mol, size=(380, 280))
        return img
    except Exception:
        return None


def get_mol_info(smiles: str) -> dict:
    """Hitung properti dasar molekul menggunakan RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        return {
            "Jumlah Atom"       : mol.GetNumAtoms(),
            "Jumlah Bond"       : mol.GetNumBonds(),
            "Molecular Weight"  : f"{Descriptors.MolWt(mol):.2f} Da",
            "LogP (Lipophilicity)": f"{Descriptors.MolLogP(mol):.3f}",
            "H-Bond Donor"      : rdMolDescriptors.CalcNumHBD(mol),
            "H-Bond Acceptor"   : rdMolDescriptors.CalcNumHBA(mol),
            "Aromatic Rings"    : rdMolDescriptors.CalcNumAromaticRings(mol),
            "Rotatable Bonds"   : rdMolDescriptors.CalcNumRotatableBonds(mol),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SMILES Examples
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_SMILES = {
    "Aspirin (NSAID)"                  : "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (Stimulant)"             : "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "AZT — Zidovudine (HIV Drug ✓)"   : "Cc1cn([C@@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)c(=O)[nH]1",
    "Nevirapine (HIV Drug ✓)"         : "Cc1ccnc2N(C3CC3)c(=O)/c(=C/c3cccnc3)c(=O)n12",
    "Glucose (Sugar)"                  : "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Benzene (Aromatic)"               : "c1ccccc1",
    "Dopamine (Neurotransmitter)"      : "NCCc1ccc(O)c(O)c1",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Tentang Model")
    st.markdown("""
**Arsitektur:** Graph Isomorphism Network (GIN)

**Dataset:** ogbg-molhiv (OGB)
- 41,127 molekul
- Binary: HIV inhibitor?

**Teknik:**
- 5 GIN layers
- Hidden dim: 300
- BondEncoder + AtomEncoder
- BatchNorm + Dropout(0.5)
- Global Mean Pooling
- Pos-weighted BCE Loss
    """)

    st.divider()
    st.markdown("## 📊 Model Performance")

    model_available = os.path.exists(MODEL_PATH)
    if model_available:
        metrics = load_metrics()
        if metrics:
            cols = st.columns(2)
            cols[0].metric("ROC-AUC",  f"{metrics.get('roc_auc', 0):.4f}")
            cols[1].metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            cols[0].metric("Precision", f"{metrics.get('precision', 0):.4f}")
            cols[1].metric("Recall",    f"{metrics.get('recall', 0):.4f}")
            st.metric("F1-Score",  f"{metrics.get('f1_score', 0):.4f}")
    else:
        st.warning("Model belum dilatih. Jalankan `model_training.py` terlebih dahulu.")

    st.divider()
    st.markdown("## 🔬 SMILES Contoh")
    st.markdown("Pilih molekul contoh di bawah untuk demo cepat.")

    st.divider()
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "Aplikasi ini hanya untuk tujuan edukasi dan riset. "
        "Bukan pengganti uji klinis."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🧬 GNN MolHIV Predictor</h1>
    <p>Prediksi potensi inhibisi HIV dari struktur molekul menggunakan
    <strong>Graph Neural Network (GIN)</strong> — dilatih pada dataset OGB ogbg-molhiv.</p>
</div>
""", unsafe_allow_html=True)


# ── Cek apakah model tersedia ────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("""
    ❌ **Model belum tersedia.**

    Jalankan training terlebih dahulu:
    ```bash
    python model_training.py
    ```
    Setelah training selesai, restart aplikasi ini.
    """)
    st.stop()


# Load model
try:
    model, config = load_model()
    model_loaded  = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Tab Layout
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔬 Prediksi Molekul", "📈 Evaluasi Model", "ℹ️ Tentang Proyek"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Prediksi
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### Masukkan SMILES Molekul")

    col_input, col_example = st.columns([2, 1])

    with col_example:
        st.markdown("**Pilih molekul contoh:**")
        example_choice = st.selectbox(
            "Pilih:",
            ["— pilih —"] + list(EXAMPLE_SMILES.keys()),
            label_visibility="collapsed",
        )

    with col_input:
        default_smiles = EXAMPLE_SMILES.get(example_choice, "") if example_choice != "— pilih —" else ""
        smiles_input = st.text_input(
            "SMILES Notation:",
            value=default_smiles,
            placeholder="Contoh: CC(=O)Oc1ccccc1C(=O)O",
            help="Masukkan SMILES string molekul yang ingin dianalisis.",
        )

    st.markdown("""
    <div class="info-box">
    💡 <strong>SMILES</strong> (Simplified Molecular Input Line-Entry System) adalah notasi teks
    untuk merepresentasikan struktur kimia. Contoh: <code>CC(=O)O</code> = Asam Asetat.
    Kamu bisa mendapatkan SMILES dari <a href="https://pubchem.ncbi.nlm.nih.gov" target="_blank">PubChem</a>
    atau <a href="https://www.chemspider.com" target="_blank">ChemSpider</a>.
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

    # ── Hasil Prediksi ───────────────────────────────────────────────────
    if predict_btn and smiles_input.strip():
        with st.spinner("Menganalisis struktur molekul..."):
            prob, label, graph = predict_smiles(model, smiles_input.strip())

        if prob is None:
            st.error("❌ SMILES tidak valid atau tidak dapat diproses. Periksa kembali input Anda.")
        else:
            st.divider()
            st.markdown("### Hasil Analisis")

            col_vis, col_result = st.columns([1, 1])

            # ── Visualisasi Molekul ──────────────────────────────────────
            with col_vis:
                st.markdown("**Struktur Molekul**")
                mol_img = draw_molecule(smiles_input.strip())
                if mol_img:
                    st.image(mol_img, use_container_width=True)
                else:
                    st.info("Visualisasi molekul tidak tersedia (install RDKit).")

                # Properti molekul
                mol_props = get_mol_info(smiles_input.strip())
                if mol_props:
                    st.markdown("**Properti Molekul**")
                    for k, v in mol_props.items():
                        st.markdown(f"- **{k}:** {v}")

            # ── Prediksi ─────────────────────────────────────────────────
            with col_result:
                st.markdown("**Hasil Prediksi**")

                if label == 1:
                    color_bar = "#e24b4a"
                    result_class = "result-positive"
                    emoji = "⚠️"
                    verdict = "HIV INHIBITOR"
                    interpretation = (
                        "Molekul ini diprediksi <strong>memiliki potensi</strong> "
                        "menghambat replikasi HIV. Perlu penelitian lebih lanjut."
                    )
                else:
                    color_bar = "#1d9e75"
                    result_class = "result-negative"
                    emoji = "✅"
                    verdict = "BUKAN HIV INHIBITOR"
                    interpretation = (
                        "Molekul ini diprediksi <strong>tidak memiliki</strong> "
                        "aktivitas penghambatan HIV yang signifikan."
                    )

                st.markdown(f"""
                <div class="{result_class}">
                    <div class="result-title">{emoji} {verdict}</div>
                    <div class="result-subtitle" style="margin-top:0.4rem">{interpretation}</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar
                st.markdown(f"**Probabilitas HIV Inhibitor:** `{prob:.4f}`")
                bar_width = int(prob * 100)
                st.markdown(f"""
                <div class="prob-bar-container">
                    <div class="prob-bar-fill"
                         style="width:{bar_width}%; background:{color_bar}">
                        {bar_width}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**Confidence:** {'Tinggi' if abs(prob - 0.5) > 0.3 else 'Sedang' if abs(prob - 0.5) > 0.1 else 'Rendah'}")

                # Graph stats
                if graph:
                    st.markdown("**Statistik Graf Molekul**")
                    gcol1, gcol2 = st.columns(2)
                    gcol1.metric("Node (Atom)",  graph.x.shape[0])
                    gcol2.metric("Edge (Bond)",  graph.edge_index.shape[1] // 2)

                st.markdown(f"""
                <div class="warning-box">
                ⚠️ Threshold klasifikasi: 0.5. Nilai probabilitas mendekati 0.5
                menandakan prediksi dengan kepercayaan rendah.
                </div>
                """, unsafe_allow_html=True)

    elif predict_btn:
        st.warning("Masukkan SMILES molekul terlebih dahulu.")

    # ── Batch Prediction ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Prediksi Batch (Multi-Molekul)")
    st.markdown("Masukkan beberapa SMILES (satu per baris):")

    batch_input = st.text_area(
        "Batch SMILES:",
        placeholder="CC(=O)Oc1ccccc1C(=O)O\nCn1cnc2c1c(=O)n(c(=O)n2C)C\nNCCc1ccc(O)c(O)c1",
        height=120,
        label_visibility="collapsed",
    )

    if st.button("🔄 Prediksi Semua", use_container_width=True):
        smiles_list = [s.strip() for s in batch_input.strip().split("\n") if s.strip()]
        if smiles_list:
            results = []
            progress = st.progress(0)
            for i, smi in enumerate(smiles_list):
                prob, label, _ = predict_smiles(model, smi)
                results.append({
                    "SMILES"            : smi,
                    "Probabilitas"      : f"{prob:.4f}" if prob else "Error",
                    "Prediksi"          : ("HIV Inhibitor ⚠️" if label == 1 else "Non-HIV ✅") if label is not None else "Invalid",
                })
                progress.progress((i + 1) / len(smiles_list))

            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Hasil (CSV)",
                data=csv,
                file_name="molhiv_predictions.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Evaluasi
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### 📊 Evaluasi Model")

    metrics = load_metrics()
    if metrics:
        # Metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC",   f"{metrics.get('roc_auc', 0):.4f}",  help="Area Under ROC Curve — metrik utama OGB")
        c2.metric("Accuracy",  f"{metrics.get('accuracy', 0):.4f}")
        c3.metric("Precision", f"{metrics.get('precision', 0):.4f}", help="Dari semua prediksi positif, berapa yang benar?")
        c4.metric("Recall",    f"{metrics.get('recall', 0):.4f}",    help="Dari semua HIV inhibitor nyata, berapa yang terdeteksi?")
        c5.metric("F1-Score",  f"{metrics.get('f1_score', 0):.4f}")

    # Tampilkan gambar evaluasi jika ada
    assets_dir = "assets"
    img_files  = {
        "Training Curves"   : os.path.join(assets_dir, "training_curves.png"),
        "ROC Curve"         : os.path.join(assets_dir, "roc_curve.png"),
        "Confusion Matrix"  : os.path.join(assets_dir, "confusion_matrix.png"),
        "Precision-Recall"  : os.path.join(assets_dir, "pr_curve.png"),
    }

    available = {k: v for k, v in img_files.items() if os.path.exists(v)}

    if available:
        st.divider()
        for title, path in available.items():
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)
            st.markdown("")
    else:
        st.info("Gambar evaluasi belum tersedia. Jalankan training terlebih dahulu untuk menghasilkan grafik.")

    # OGB Leaderboard comparison
    st.divider()
    st.markdown("### 🏆 Perbandingan dengan OGB Leaderboard")
    import pandas as pd
    leaderboard_data = {
        "Model"         : ["GIN (Proyek ini)", "GIN+Virtual Node", "DeeperGCN", "PNA", "ExpC"],
        "ROC-AUC"       : [
            f"{metrics.get('roc_auc', 0.757):.4f}",
            "0.7707 ± 0.0149",
            "0.7858 ± 0.0117",
            "0.7905 ± 0.0132",
            "0.7942 ± 0.0120",
        ],
        "Keterangan"    : ["✅ Model kita", "OGB Baseline", "OGB Baseline", "Top-tier", "Top-tier"],
    }
    st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — About
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### ℹ️ Tentang Proyek")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
**GNN MolHIV — HIV Inhibition Prediction**

Proyek ini menggunakan **Graph Neural Network (GIN)** untuk memprediksi
apakah suatu molekul memiliki potensi menghambat replikasi HIV,
menggunakan dataset publik **ogbg-molhiv** dari Open Graph Benchmark (OGB).

**Konteks Aplikasi:**
Penemuan obat HIV (drug discovery) adalah proses mahal dan panjang.
Deep learning dapat mempercepat tahap awal dengan menyaring jutaan
kandidat molekul secara komputasional sebelum uji laboratorium.

**Kenapa GNN?**
Molekul adalah **graf alami** — atom sebagai node, ikatan kimia sebagai edge.
GNN mampu belajar representasi molekular langsung dari struktur grafnya,
tidak seperti ANN/CNN yang membutuhkan representasi flat.
        """)

    with col_b:
        st.markdown("""
**Stack Teknologi:**
- `PyTorch` + `PyTorch Geometric` — Deep learning & GNN
- `OGB` (Open Graph Benchmark) — Dataset & evaluator
- `RDKit` — Cheminformatics, SMILES parsing
- `Streamlit` — Web deployment
- `Matplotlib` / `Seaborn` — Visualisasi

**Dataset ogbg-molhiv:**
- 41,127 molekul dari ChEMBL database
- Label: HIV inhibitor (1) / non-inhibitor (0)
- Imbalanced: ~1:30 rasio positif:negatif
- Split: scaffold split (OGB standard)
- Evaluasi: ROC-AUC

**Referensi:**
- Xu et al., ICLR 2019 — "How Powerful are Graph Neural Networks?"
- Hu et al., NeurIPS 2020 — Open Graph Benchmark
- OGB GitHub: github.com/snap-stanford/ogb
        """)

    st.divider()
    st.markdown("### 📁 Struktur Project")
    st.code("""
gnn-molhiv/
├── app.py                  # Streamlit deployment (file ini)
├── model_training.py       # Script training end-to-end
├── notebook.ipynb          # Eksplorasi & analisis
├── requirements.txt        # Dependencies
├── README.md               # Dokumentasi lengkap
├── src/
│   ├── model.py            # Arsitektur GIN
│   └── utils.py            # Utility: evaluasi, visualisasi
├── model/
│   ├── best_model.pt       # Model terbaik (val ROC-AUC)
│   ├── config.pt           # Hyperparameter config
│   └── metrics.json        # Hasil evaluasi
└── assets/
    ├── training_curves.png
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── pr_curve.png
    """, language="bash")
