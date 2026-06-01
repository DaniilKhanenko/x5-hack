FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates wget build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "NUM_WORKERS=${NUM_WORKERS:-$(nproc)} && echo \"NUM_WORKERS=$NUM_WORKERS\" && exec gunicorn -k uvicorn.workers.UvicornWorker main:app --workers $NUM_WORKERS --bind 0.0.0.0:8000 --timeout 120 --log-level info"]
