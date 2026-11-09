"""Smoke test: exec the notebook cells locally on tiny settings (real data, 1 stream, few steps)."""
import os, sys, time, tempfile
from pathlib import Path

os.environ["ABC_ROOT"] = r"d:\Desktop\ABC27"
HERE = Path(__file__).parent / "cells"
ns = {"__name__": "smoke"}
tmp = Path(tempfile.mkdtemp(prefix="abc_smoke_"))


def run(name):
    t = time.time()
    exec(compile((HERE / name).read_text(encoding="utf-8"), name, "exec"), ns)
    print(f"--- {name} done in {time.time() - t:.1f}s\n")


run("cell01_setup.py")
ns.update(CKPT_DIR=tmp / "ckpt", CACHE_DIR=tmp / "cache", SUBMISSION_CSV=tmp / "sub.csv",
          TRAIN_STREAM_IDS=["val_04"], VAL_STREAM_IDS=["val_04"], TEST_STREAM_IDS=["test_04"],
          S1_STEPS=int(os.environ.get("S1", 12)), S2_STEPS=int(os.environ.get("S2", 12)), S1_BATCH=int(os.environ.get("S1B", 64)))
ns["CKPT_DIR"].mkdir(parents=True); ns["CACHE_DIR"].mkdir(parents=True)
# functions defined later use CACHE_DIR as default arg -> patched by re-exec below
for n in ["cell02_preprocessing.py", "cell03_stage1.py", "cell04_stage2.py", "cell05_loss.py", "cell06_inference.py",
          "cell06b_training.py", "cell06c_validation.py", "cell07_test_submission.py"]:
    run(n)
print(ns["submission"].head())
print(tmp)
