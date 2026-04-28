#!/bin/bash
# ============================================================
#  setup.sh — Auto Setup GNN MolHIV (macOS Apple Silicon)
#  Jalankan: bash setup.sh
# ============================================================

set -e  # Stop jika ada error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  GNN MolHIV — Auto Setup Script          ${NC}"
echo -e "${BLUE}  macOS Apple Silicon (M1/M2/M3)           ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# ── Step 1: Cek Python ──────────────────────────────────────
echo -e "${YELLOW}[1/6] Mengecek Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 tidak ditemukan. Install dari https://python.org${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

# ── Step 2: Buat Virtual Environment ───────────────────────
echo ""
echo -e "${YELLOW}[2/6] Membuat virtual environment 'venv'...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Folder venv sudah ada, skip pembuatan.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment 'venv' berhasil dibuat${NC}"
fi

# ── Step 3: Aktifkan venv ───────────────────────────────────
echo ""
echo -e "${YELLOW}[3/6] Mengaktifkan virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ venv aktif: $(which python)${NC}"

# ── Step 4: Upgrade pip ────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/6] Upgrade pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip up to date${NC}"

# ── Step 5: Install semua dependencies ─────────────────────
echo ""
echo -e "${YELLOW}[5/6] Install dependencies (ini butuh 3-5 menit)...${NC}"
echo ""

echo -e "  → Installing PyTorch (CPU/MPS untuk Apple Silicon)..."
pip install torch torchvision --quiet
echo -e "${GREEN}  ✓ PyTorch${NC}"

echo -e "  → Installing PyTorch Geometric..."
pip install torch-geometric --quiet
echo -e "${GREEN}  ✓ PyTorch Geometric${NC}"

echo -e "  → Installing OGB (Open Graph Benchmark)..."
pip install ogb --quiet
echo -e "${GREEN}  ✓ OGB${NC}"

echo -e "  → Installing RDKit..."
pip install rdkit --quiet
echo -e "${GREEN}  ✓ RDKit${NC}"

echo -e "  → Installing data science & deployment libraries..."
pip install streamlit pandas numpy scikit-learn matplotlib seaborn tqdm Pillow --quiet
echo -e "${GREEN}  ✓ Streamlit, Pandas, NumPy, Sklearn, Matplotlib, Seaborn${NC}"

echo -e "  → Installing Jupyter..."
pip install jupyter ipykernel ipywidgets --quiet
echo -e "${GREEN}  ✓ Jupyter${NC}"

# Daftarkan kernel ke Jupyter
python -m ipykernel install --user --name=gnn-molhiv --display-name "Python (gnn-molhiv)"
echo -e "${GREEN}  ✓ Jupyter kernel 'gnn-molhiv' terdaftar${NC}"

# ── Step 6: Verifikasi ──────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6] Verifikasi instalasi...${NC}"
python3 - << 'PYEOF'
import sys

checks = {
    "torch"         : "import torch; print(f'PyTorch {torch.__version__}')",
    "torch_geometric": "import torch_geometric; print(f'PyG {torch_geometric.__version__}')",
    "ogb"           : "import ogb; print(f'OGB {ogb.__version__}')",
    "rdkit"         : "from rdkit import Chem; print('RDKit OK')",
    "streamlit"     : "import streamlit; print(f'Streamlit {streamlit.__version__}')",
    "pandas"        : "import pandas; print(f'Pandas {pandas.__version__}')",
    "sklearn"       : "import sklearn; print(f'Sklearn {sklearn.__version__}')",
}

all_ok = True
for name, cmd in checks.items():
    try:
        exec(cmd)
    except Exception as e:
        print(f"❌ {name}: {e}")
        all_ok = False

if all_ok:
    print("\n✅ Semua dependencies berhasil diinstall!")
else:
    print("\n⚠️  Ada library yang gagal — lihat error di atas")
    sys.exit(1)
PYEOF

# ── Selesai ─────────────────────────────────────────────────
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}  ✅ SETUP SELESAI!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Langkah selanjutnya:"
echo -e "  ${GREEN}1.${NC} Aktifkan venv   : source venv/bin/activate"
echo -e "  ${GREEN}2.${NC} Training model  : python model_training.py"
echo -e "  ${GREEN}3.${NC} Jalankan app    : streamlit run app.py"
echo -e "  ${GREEN}4.${NC} Buka notebook   : jupyter notebook notebook.ipynb"
echo ""
