"""
download_data.py
----------------
Helper script to download the UCI Online Retail dataset.
Run this once to get your data:
    python download_data.py
"""

import os
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DATA_DIR  = "data"
ZIP_PATH  = os.path.join(DATA_DIR, "online_retail.zip")
XLSX_PATH = os.path.join(DATA_DIR, "Online Retail.xlsx")

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("CUSTOMER SEGMENTATION — Dataset Setup")
print("=" * 60)
print("\n📥 Downloading UCI Online Retail dataset...")
print("   URL: https://archive.ics.uci.edu/ml/datasets/Online+Retail")
print("   Size: ~22 MB (zip) → ~44 MB (xlsx)")
print()

try:
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print("✅ Download complete!")

    import zipfile
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(DATA_DIR)
    print("✅ Extracted to data/ folder")
    os.remove(ZIP_PATH)

except Exception as e:
    print(f"⚠️  Auto-download failed: {e}")
    print()
    print("Please download manually:")
    print("1. Go to: https://archive.ics.uci.edu/ml/datasets/Online+Retail")
    print("2. Click 'Download' → save 'Online Retail.xlsx'")
    print(f"3. Place it at: {os.path.abspath(XLSX_PATH)}")
    print()
    print("OR: The notebook will auto-generate synthetic data for demo purposes.")

print()
print("Next steps:")
print("  pip install -r requirements.txt")
print("  jupyter notebook notebooks/customer_segmentation_analysis.ipynb")
print("  streamlit run streamlit_app.py")