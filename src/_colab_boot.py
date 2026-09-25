
"""Idempotent Colab session bootstrap."""
import os, sys, subprocess

DATA_ROOT   = '/content/drive/MyDrive/amazon_ml'
DATASET_DIR = f'{DATA_ROOT}/dataset'
TRAIN_DIR   = f'{DATASET_DIR}/train'
TEST_DIR    = f'{DATASET_DIR}/test'
OUT_DIR     = f'{DATA_ROOT}/work'
REPO_DIR    = f'{DATA_ROOT}/repo'

REQUIRED_PKGS = [
    'duckdb', 'rapidfuzz', 'jellyfish', 'sentence_transformers',
    'lightgbm', 'indic_transliteration', 'datasketch',
]
PIP_NAMES = {
    'sentence_transformers': 'sentence-transformers',
    'indic_transliteration': 'indic-transliteration',
}


def _mount_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception as e:
        print(f"  [warn] Drive mount: {e}")


def _install_packages():
    import importlib
    missing = []
    for pkg in REQUIRED_PKGS:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        pkgs = ' '.join(PIP_NAMES.get(m, m) for m in missing)
        print(f"  installing: {pkgs}")
        subprocess.run(f"pip install -q {pkgs}", shell=True, check=True)
    try:
        importlib.import_module('faiss')
    except ImportError:
        print("  installing: faiss-cpu")
        subprocess.run("pip install -q faiss-cpu", shell=True, check=True)


def _git_auth():
    try:
        from google.colab import userdata
        tok = userdata.get('GITHUB_TOKEN')
        if not tok:
            raise RuntimeError("GITHUB_TOKEN secret is empty")
        with open("/root/.git-credentials", "w") as f:
            f.write(f"https://varanasi-ramya:{tok}@github.com\n")
        os.chmod("/root/.git-credentials", 0o600)
        subprocess.run("git config --global credential.helper store",
                       shell=True, check=True)
    except Exception as e:
        print(f"  [warn] Git auth: {e}")
    subprocess.run('git config --global user.name  "varanasi-ramya"',
                   shell=True, check=True)
    subprocess.run('git config --global user.email '
                   '"nvsriramyavaranasi@gmail.com"',
                   shell=True, check=True)


def boot(verbose=True):
    _mount_drive()
    for d in (OUT_DIR, REPO_DIR, f'{REPO_DIR}/src', f'{REPO_DIR}/notebooks'):
        os.makedirs(d, exist_ok=True)
    _install_packages()
    _git_auth()
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    import torch
    gpu = torch.cuda.is_available()
    ctx = dict(
        DATA_ROOT=DATA_ROOT, DATASET_DIR=DATASET_DIR,
        TRAIN_DIR=TRAIN_DIR, TEST_DIR=TEST_DIR,
        OUT_DIR=OUT_DIR, REPO_DIR=REPO_DIR, gpu=gpu,
    )
    if verbose:
        dev = torch.cuda.get_device_name(0) if gpu else 'CPU'
        print(f"Boot OK | GPU: {gpu} ({dev}) | REPO: {REPO_DIR}")
    return ctx
