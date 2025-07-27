FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgl1-mesa-glx \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install nltk \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Pre-download MiniLM model
RUN mkdir -p /models && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/models/all-MiniLM-L6-v2')"


# Pre-download NLTK punkt data
RUN python -m nltk.downloader punkt punkt_tab


COPY . .

CMD ["python", "adobe_1b.py", "Collection 1"]
