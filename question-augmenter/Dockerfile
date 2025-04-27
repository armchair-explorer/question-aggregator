FROM python:3.9-slim

WORKDIR /app


RUN apt-get update && \
    apt-get install -y \
        build-essential \
        python3-dev \
        libpython3-dev \
        libopenblas-dev \
        g++ && \
    pip install --no-cache-dir \
        "numpy<2.0.0" \
        sentence-transformers==2.2.2 \
        transformers==4.28.1 \
        torch==1.13.1 \
        neo4j \
        tqdm \
        scikit-learn \
        hdbscan \
        faiss-cpu \
        scikit-learn \
        huggingface_hub==0.14.1

COPY . /app

CMD ["python3", "vectorize_and_push.py"]

