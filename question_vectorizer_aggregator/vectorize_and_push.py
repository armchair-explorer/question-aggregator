import os
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import faiss
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import csv

# Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testtest")
QUESTIONS_FILE = "chapter2_os_questions_cleaned.txt"
SIMILARITY_THRESHOLD = 0.85
BATCH_SIZE = 1000
PARALLEL_JOBS = 4

# 🔁 Load questions
with open(QUESTIONS_FILE, "r") as file:
    questions = [line.strip() for line in file if line.strip()]
print(f"📚 Loaded {len(questions)} questions.")

# 🔍 Embeddings with context-aware model
model = SentenceTransformer("intfloat/e5-large-v2")

embeddings = model.encode(questions, normalize_embeddings=True).astype('float32')
N, D = embeddings.shape

# 🎯 Build FAISS index for similarity
print("🎯 Building FAISS index...")
index = faiss.IndexFlatIP(D)
index.add(embeddings)
k = 50
scores, neighbors = index.search(embeddings, k)

# 🔌 Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 🧠 Insert question node (without cluster id initially)
def insert_node(tx, idx, text, embedding):
    tx.run("""
        CREATE (q:Question {id: $id, text: $text, embedding: $embedding})
    """, id=idx, text=text, embedding=embedding.tolist())

# 🧠 Batch insert similarity edges
def insert_edges_batch(tx, edge_batch):
    tx.run("""
        UNWIND $batch as row
        MATCH (a:Question {id: row.src}), (b:Question {id: row.tgt})
        MERGE (a)-[:SIMILAR {score: row.score}]->(b)
    """, batch=edge_batch)

# ✅ Insert nodes
print("📌 Inserting question nodes...")
with driver.session() as session:
    for i, (q, emb) in enumerate(tqdm(zip(questions, embeddings), total=N)):
        session.write_transaction(insert_node, i, q, emb)

# 🔗 Prepare edge batches
print("🔍 Querying neighbors above threshold...")
edge_batches = [[] for _ in range(PARALLEL_JOBS)]

def build_edges(start, end, thread_id):
    for i in range(start, end):
        for j, score in zip(neighbors[i], scores[i]):
            if i != j and score >= SIMILARITY_THRESHOLD:
                edge_batches[thread_id].append({
                    "src": int(i),
                    "tgt": int(j),
                    "score": float(score)
                })

# 🚀 Parallel similarity filtering
print("⚡ Filtering & batching edges...")
chunk_size = N // PARALLEL_JOBS
with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as executor:
    for t in range(PARALLEL_JOBS):
        start = t * chunk_size
        end = N if t == PARALLEL_JOBS - 1 else (t + 1) * chunk_size
        executor.submit(build_edges, start, end, t)

# 🧱 Insert edge batches
print("🚚 Inserting similarity edges into Neo4j...")
with driver.session() as session:
    for batch in tqdm(edge_batches):
        for i in range(0, len(batch), BATCH_SIZE):
            chunk = batch[i:i + BATCH_SIZE]
            session.write_transaction(insert_edges_batch, chunk)

# 🧠 Clustering using HDBSCAN (AFTER graph construction)
print("🔗 Clustering with HDBSCAN (cosine)...")
#clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean', cluster_selection_epsilon=0.4)
clusterer = AgglomerativeClustering(
    n_clusters=None,
    metric='cosine',
    linkage='average',
    distance_threshold=0.12
)

cluster_ids = clusterer.fit_predict(embeddings)
print(f"📦 Total clusters found: {len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)}")

# ⛓️ Update cluster ID in Neo4j
def insert_cluster_id(tx, idx, cluster_id):
    tx.run("MATCH (q:Question {id: $id}) SET q.cluster_id = $cid", id=idx, cid=int(cluster_id) if cluster_id != -1 else None)

print("📥 Inserting cluster IDs...")
with driver.session() as session:
    for i, cid in enumerate(tqdm(cluster_ids)):
        session.write_transaction(insert_cluster_id, i, cid)

output_path = "clustered_questions.csv"

# Write to CSV
with open(output_path, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "ClusterID"])
    for question, cid in zip(questions, cluster_ids):
        writer.writerow([question, cid])

print(f"📁 Saved clustered questions to: {output_path}")


print("✅ Done inserting nodes + edges + clusters.")

