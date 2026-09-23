# Azriel. Builds the page and the API into one image; which one runs is the
# command, so the same image serves both.
FROM python:3.11-slim

# curl for the healthcheck; the rest is what pymupdf and onnxruntime link to.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing a source file does not reinstall 700 MB.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding model into the image. Without this the first question
# after every deploy waits on a 90 MB download, and a container with no
# network to HuggingFace never answers at all.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Documents live here. Mount a volume on it or they go when the container is
# replaced - which for the page's visitor collections is fine, and for the
# API's named collections is not.
RUN mkdir -p /app/data
ENV AZRIEL_COLLECTIONS=visitor \
    AZRIEL_PUBLIC=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# The page. For the API instead:
#   docker run ... azriel python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
