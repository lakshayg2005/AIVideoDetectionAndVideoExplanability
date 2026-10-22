"""Embed the current code (forensics/, explain/face_parts.py, M2F2Det) into a Kaggle job template, so the job
cannot run with a stale dataset version.

python kaggle/make_kernel.py kaggle/stagea_v4.py <kernel_dir>/stagea_v4.py
"""
import base64
import io
import sys
import tarfile

FILES = ['forensics', 'explain/__init__.py', 'explain/face_parts.py', 'llava/model/deepfake/M2F2Det/model.py',
         'llava/model/deepfake/M2F2Det/text_encoder.py', 'llava/model/deepfake/M2F2Det/vision_encoder.py']

if __name__ == '__main__':
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        for f in FILES:
            t.add(f, filter=lambda ti: None if '__pycache__' in ti.name else ti)
    src = open(sys.argv[1]).read()
    assert src.count("'__CODE_B64__'") == 1
    open(sys.argv[2], 'w').write(src.replace("'__CODE_B64__'", repr(base64.b64encode(buf.getvalue()).decode())))
    print(f'embedded {len(buf.getvalue())} bytes of code into {sys.argv[2]}')
