# The agent's Python kernel: CPU only, no network at run time. No transformers, no mlx, no HF cache.
FROM python:3.12-slim
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir numpy scipy scikit-learn pandas sympy networkx safetensors tokenizers \
 && pip freeze > /requirements.lock \
 && pip uninstall -y pip setuptools wheel || true
RUN useradd -u 1000 -d /tmp -M -s /usr/sbin/nologin kernel
USER 1000:1000
WORKDIR /work
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
