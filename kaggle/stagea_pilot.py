"""Kaggle job: Stage-A pilot (detector + mask head + type/filter head) on a T4.

Inputs: the private code dataset (forensics/, explain/, M2F2Det code, DDVQA c40) and the CASIA v2
Kaggle dataset. Downloaded here: M2F2 checkpoint shard 3 (vision tower + detector), CASIA ground
truth, SD_inpaint, OpenForensics Val. Outputs in /kaggle/working/stagea (log.jsonl, last.pt).
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

T0 = time.time()
DATA = '/tmp/data'
OUT = '/kaggle/working/stagea'
os.makedirs(DATA, exist_ok=True)


def sh(cmd):
    print('+', cmd, flush=True)
    subprocess.run(['/bin/bash', '-o', 'pipefail', '-c', cmd], check=True)


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


def fetch(url, dst):
    if not os.path.exists(dst):
        log(f'download {url}')
        urllib.request.urlretrieve(url, dst)
    return dst


if sys.version_info[:2] != (3, 10):
    # Kaggle's image is newer than the pinned stack (transformers 4.37, mediapipe 0.10.14): rebuild the
    # local Python 3.10 environment with uv and re-run this script inside it
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /tmp /kaggle/working | tail -2')
    venv = '/tmp/venv'
    sh(f'{sys.executable} -m pip install -q uv && uv venv -q -p 3.10 {venv}')
    sh(f'uv pip install -q -p {venv}/bin/python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121')
    sh(f'uv pip install -q -p {venv}/bin/python numpy==1.26.4 transformers==4.37.0 tokenizers==0.15.1 safetensors==0.4.3 '
       f'huggingface_hub==0.22.2 mediapipe==0.10.14 opencv-python==4.8.1.78 pillow')
    os.execv(f'{venv}/bin/python', [f'{venv}/bin/python', os.path.abspath(__file__)])

# code: the dataset may be mounted as folders or as an archive
code_src = next(iter(glob.glob('/kaggle/input/**/forensics/model.py', recursive=True)), None)
CODE = '/kaggle/working/code'
if code_src:
    shutil.copytree(os.path.dirname(os.path.dirname(code_src)), CODE, dirs_exist_ok=True)
else:
    arc = next(iter(glob.glob('/kaggle/input/**/code.tar.gz', recursive=True)))
    tarfile.open(arc).extractall(CODE)
sys.path.insert(0, CODE)
os.chdir(CODE)
log(f'code at {CODE}')

# M2F2 checkpoint shard with the vision tower and the detector
from huggingface_hub import hf_hub_download  # noqa: E402
shard = hf_hub_download('CHELSEA234/llava-v1.5-7b-M2F2-Det', 'model-00003-of-00003.safetensors', local_dir=f'{DATA}/m2f2')
log('m2f2 shard ready')

# CASIA v2: Kaggle input (read-only) + revised ground truth from GitHub
tp = next(iter(glob.glob('/kaggle/input/**/CASIA2/Tp', recursive=True)))
casia = f'{DATA}/casia'
os.makedirs(casia, exist_ok=True)
if not os.path.exists(f'{casia}/CASIA2'):
    os.symlink(os.path.dirname(tp), f'{casia}/CASIA2')
gtz = fetch('https://raw.githubusercontent.com/namtpham/casia2groundtruth/master/CASIA2.0_Groundtruth.zip', f'{DATA}/casia_gt.zip')
zipfile.ZipFile(gtz).extractall(f'{casia}/gt')
log('casia ready')

# SD_inpaint (FakeShield)
sd = f'{DATA}/sd_inpaint'
for part in ('eval', 'train'):
    if not os.path.isdir(f'{sd}/{part}'):
        t = hf_hub_download('zhipeixu/SD_inpaint_dataset', f'{part}.tar', repo_type='dataset', local_dir=f'{DATA}/sd_tar')
        tarfile.open(t).extractall(sd)
        os.remove(t)
log('sd_inpaint ready')

# OpenForensics Val (Zenodo, CC-BY-4.0): images + polygons, split 80/20 by image for the pilot
of = f'{DATA}/openforensics'
os.makedirs(of, exist_ok=True)
fetch('https://zenodo.org/records/5528418/files/Val_poly.json?download=1', f'{of}/Val_poly.json')
if not os.path.isdir(f'{of}/Images/Val'):
    z = fetch('https://zenodo.org/records/5528418/files/Val.zip?download=1', f'{DATA}/of_val.zip')
    zipfile.ZipFile(z).extractall(of)
    os.remove(z)
if not os.path.isdir(f'{of}/Images/Val'):    # the json expects Images/Val/<name>.jpg
    img_dir = os.path.dirname(next(iter(glob.glob(f'{of}/**/*.jpg', recursive=True))))
    os.makedirs(f'{of}/Images', exist_ok=True)
    os.symlink(img_dir, f'{of}/Images/Val')
    log(f'OpenForensics images found in {img_dir}, linked as Images/Val')
log('openforensics ready')
sh('df -h /tmp | tail -1')

IDX = f'{DATA}/index'
py = sys.executable
sh(f'{py} -m forensics.index casia --root {casia} --out {IDX}/casia.jsonl')
sh(f'{py} -m forensics.index sd_inpaint --root {sd} --out {IDX}/sd_inpaint.jsonl')
sh(f'{py} -m forensics.index ddvqa --root {CODE}/utils --out {IDX}/ddvqa_c40.jsonl')
sh(f'{py} -m forensics.index openforensics --root {of} --poly-json Val_poly.json --val-frac 0.2 --out {IDX}/openforensics.jsonl')

budget = 11 * 60 - (time.time() - T0) / 60 - 40    # leave time for the final eval and save
sh(f'{py} -W ignore -m forensics.train --index {IDX}/casia.jsonl {IDX}/sd_inpaint.jsonl {IDX}/ddvqa_c40.jsonl {IDX}/openforensics.jsonl '
   f'--roots casia={casia} sd_inpaint={sd} ddvqa={CODE}/utils openforensics={of} --m2f2-shard {shard} --out {OUT} '
   f'--steps 8000 --bs 12 --eval-every 2000 --eval-at-start --workers 4 --max-minutes {budget:.0f} '
   f'2>&1 | grep -v -E "^(I0000|W0000)|absl::|XNNPACK|feedback"')
log('done')
