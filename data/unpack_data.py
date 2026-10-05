"""Rebuild elliptic_txs_features.csv from its compressed parts.

GitHub does not accept files larger than 100 MB, and the features file is about 370 MB,
so it is stored here gzip-compressed and split into four parts. Run once before the notebook:

    python data/unpack_data.py
"""
import glob
import gzip
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "elliptic_bitcoin_dataset")
TARGET = os.path.join(FOLDER, "elliptic_txs_features.csv")

parts = sorted(glob.glob(os.path.join(FOLDER, "elliptic_txs_features.csv.gz.part*")))
if os.path.exists(TARGET):
    print("Already unpacked:", TARGET)
elif not parts:
    raise SystemExit("No parts found in " + FOLDER)
else:
    gz_path = TARGET + ".gz"
    with open(gz_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)
    with gzip.open(gz_path, "rb") as f_in, open(TARGET, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"Unpacked {len(parts)} parts -> {TARGET} ({os.path.getsize(TARGET) / 1e6:.0f} MB)")
