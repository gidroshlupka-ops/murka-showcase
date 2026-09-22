FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir "pip<24.1" setuptools wheel

COPY voice/requirements.txt requirements.txt

RUN pip install --no-cache-dir "numpy==1.23.5"
RUN pip install --no-cache-dir -r requirements.txt --use-deprecated=legacy-resolver

COPY voice/app.py app.py
COPY weights weights

EXPOSE 7860
ENV PORT=7860
ENV RVC_PITCH=6

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
