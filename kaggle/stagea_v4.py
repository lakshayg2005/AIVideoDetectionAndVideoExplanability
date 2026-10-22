"""Kaggle job: Stage-A v4 on a T4 (template: kaggle/make_kernel.py embeds the code below).

Changes from the pilot (RESULTS.md section 3): all of OpenForensics with its official splits (Kaggle copy
nathanielescuro/veritas-openforensics-compiled), DiffusionFace (fully generated, img2img, inpainting,
diffusion face swap; LDM and SD 2.1 text2img held out as unseen generators), distillation from the
released detector on DDVQA, false-positive rate on filtered real images. Outputs: /kaggle/working/stagea.
v4: DiffusionFace index fix (Inpaint masks as pixel masks, mask / metadata files are not samples). The code is
embedded in this script (v3 silently ran with a stale dataset version) and the index is checked before training.
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
import base64
import io

CODE_B64 = '__CODE_B64__'    # filled in by kaggle/make_kernel.py
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
DATA = '/tmp/data'
OUT = '/kaggle/working/stagea'
os.makedirs(DATA, exist_ok=True)


def sh(cmd):
    print('+', cmd, flush=True)
    subprocess.run(['/bin/bash', '-o', 'pipefail', '-c', cmd], check=True)


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


def fetch(url, dst, tries=5):
    for i in range(tries):
        try:
            if not os.path.exists(dst):
                urllib.request.urlretrieve(url, dst + '.part')
                os.rename(dst + '.part', dst)
            return dst
        except Exception as e:    # Zenodo resets connections now and then
            log(f'retry {i + 1} {url}: {e}')
            time.sleep(20)
    raise RuntimeError(url)


if sys.version_info[:2] != (3, 10):
    # Kaggle's image is newer than the pinned stack (transformers 4.37, mediapipe 0.10.14): rebuild the
    # local Python 3.10 environment with uv and re-run this script inside it
    sh('nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2')
    venv = '/tmp/venv'
    sh(f'{sys.executable} -m pip install -q uv && uv venv -q -p 3.10 {venv}')
    sh(f'uv pip install -q -p {venv}/bin/python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121')
    sh(f'uv pip install -q -p {venv}/bin/python numpy==1.26.4 transformers==4.37.0 tokenizers==0.15.1 safetensors==0.4.3 '
       f'huggingface_hub==0.22.2 mediapipe==0.10.14 opencv-python==4.8.1.78 pillow')
    os.execv(f'{venv}/bin/python', [f'{venv}/bin/python', os.path.abspath(__file__)])

# code: embedded (forensics/, explain/, M2F2Det); DDVQA utils/ from the dataset archive
CODE = '/tmp/code'
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64))).extractall(CODE)
# Kaggle may mount the dataset archive extracted (folders) or as code.tar.gz
split_json = next(iter(glob.glob('/kaggle/input/**/utils/DDVQA_split/c40/train_DDVQA_format.json', recursive=True)), None)
if split_json:
    shutil.copytree(split_json.rsplit('/DDVQA_split/', 1)[0], f'{CODE}/utils', dirs_exist_ok=True)
else:
    with tarfile.open(next(iter(glob.glob('/kaggle/input/**/code.tar.gz', recursive=True)))) as t:
        t.extractall(CODE, members=[m for m in t.getmembers() if m.name.startswith('utils/')])
sys.path.insert(0, CODE)
os.chdir(CODE)


def get_diffusionface():
    """Download the DiffusionFace subsets in parallel (Zenodo throttles single connections)."""
    df = f'{DATA}/diffusionface'
    want = {'10865300': ['Real', 'ADM', 'SDv15_DS0.5', 'Inpaint', 'DiffSwap', 'LDM'],
            '10865065': ['stable_diffusion_v_1_5_text2img_p4g5', 'stable_diffusion_v_2_1_text2img_p1g7']}

    def one(job):
        rec, name = job
        t = fetch(f'https://zenodo.org/records/{rec}/files/{name}.tar?download=1', f'{DATA}/{name}.tar')
        tarfile.open(t).extractall(f'{df}/{name}')    # folder named after the tar = generator name
        os.remove(t)
        return name

    with ThreadPoolExecutor(8) as ex:
        for name in ex.map(one, [(r, n) for r, ns in want.items() for n in ns]):
            log(f'diffusionface {name} ready')
    return df


with ThreadPoolExecutor(2) as pool:    # DiffusionFace (slow, Zenodo) in parallel with the rest
    df_future = pool.submit(get_diffusionface)

    from huggingface_hub import hf_hub_download  # noqa: E402
    shard = hf_hub_download('CHELSEA234/llava-v1.5-7b-M2F2-Det', 'model-00003-of-00003.safetensors', local_dir=f'{DATA}/m2f2')
    log('m2f2 shard ready')

    tp = next(iter(glob.glob('/kaggle/input/**/CASIA2/Tp', recursive=True)))
    casia = f'{DATA}/casia'
    os.makedirs(casia, exist_ok=True)
    if not os.path.exists(f'{casia}/CASIA2'):
        os.symlink(os.path.dirname(tp), f'{casia}/CASIA2')
    zipfile.ZipFile(fetch('https://raw.githubusercontent.com/namtpham/casia2groundtruth/master/CASIA2.0_Groundtruth.zip',
                          f'{DATA}/casia_gt.zip')).extractall(f'{casia}/gt')
    log('casia ready')

    sd = f'{DATA}/sd_inpaint'
    for part in ('eval', 'train'):
        if not os.path.isdir(f'{sd}/{part}'):
            t = hf_hub_download('zhipeixu/SD_inpaint_dataset', f'{part}.tar', repo_type='dataset', local_dir=f'{DATA}/sd_tar')
            tarfile.open(t).extractall(sd)
            os.remove(t)
    log('sd_inpaint ready')

    of = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/compiled_poly.json', recursive=True))))
    log(f'openforensics at {of}')
    df = df_future.result()

IDX = f'{DATA}/index'
py = sys.executable
sh(f'{py} -m forensics.index casia --root {casia} --out {IDX}/casia.jsonl')
sh(f'{py} -m forensics.index sd_inpaint --root {sd} --out {IDX}/sd_inpaint.jsonl')
sh(f'{py} -m forensics.index ddvqa --root {CODE}/utils --out {IDX}/ddvqa_c40.jsonl')
sh(f'{py} -m forensics.index openforensics --root {of} --poly-json compiled_poly.json --out {IDX}/openforensics.jsonl')
sh(f'{py} -m forensics.index diffusionface --root {df} --out {IDX}/diffusionface.jsonl')
import json  # noqa: E402
dfr = [json.loads(l) for l in open(f'{IDX}/diffusionface.jsonl')]
n_inp = sum(r['image'].startswith('Inpaint') for r in dfr)
bad = [r['image'] for r in dfr if '/masks/' in r['image'] or '/jsons/' in r['image']]
log(f'diffusionface check: {len(dfr)} records, Inpaint {n_inp}, mask/metadata samples {len(bad)}')
assert not bad and n_inp == 30000, 'stale DiffusionFace index'

budget = 11.5 * 60 - (time.time() - T0) / 60 - 45    # leave time for the final eval and save
idx = ' '.join(f'{IDX}/{n}.jsonl' for n in ('casia', 'sd_inpaint', 'ddvqa_c40', 'openforensics', 'diffusionface'))
sh(f'{py} -W ignore -m forensics.train --index {idx} '
   f'--roots casia={casia} sd_inpaint={sd} ddvqa={CODE}/utils openforensics={of} diffusionface={df} '
   f'--m2f2-shard {shard} --out {OUT} --steps 14000 --bs 12 --eval-every 3500 --eval-at-start --workers 4 '
   f'--distill 1.0 --max-minutes {budget:.0f} 2>&1 | grep -v -E "^(I0000|W0000)|absl::|XNNPACK|feedback"')
log('done')
