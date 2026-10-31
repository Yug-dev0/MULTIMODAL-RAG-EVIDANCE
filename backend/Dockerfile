FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg tesseract-ocr libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU-only torch/torchaudio first — pip's default index pulls the CUDA build
# (multiple GB of unused NVIDIA libraries) otherwise; pyannote.audio then
# reuses this already-installed CPU torch instead of pulling GPU wheels.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000
