import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline,pipeline, AutoModelForCausalLM
from neo4j import GraphDatabase
from collections import defaultdict
from tqdm import tqdm
import psycopg2


# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testtest")
OUTPUT_FILE = "cluster_augmented_questions7.txt"

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model/tokenizer manually to avoid pipeline bugs
#tokenizer = AutoTokenizer.from_pretrained("t5-small")
#model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)

#generator = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0 if device.type == "mps" else -1)


generator = pipeline(
    "text2text-generation",

    model="google/long-t5-tglobal-base",
    tokenizer="google/long-t5-tglobal-base"
#  model="sshleifer/distilmbart-cnn-6-6", tokenizer="sshleifer/distilmbart-cnn-6-6"
)


# Postgres connection
conn = psycopg2.connect(
    host="postgres",
    database="qa_db",
    user="user",
    password="password"
)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        classid INT,
        clusterid INT,
        questionid SERIAL PRIMARY KEY ,
        question TEXT,
        answer TEXT
    );
""")
conn.commit()

cur.execute("""
    CREATE TABLE IF NOT EXISTS og_questions (
        classid INT,
        clusterid INT,
        questionid SERIAL PRIMARY KEY ,
        question TEXT,
        answer TEXT
    );
""")
conn.commit()



# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def fetch_clusters(tx):
    result = tx.run("""
        MATCH (q:Question)
        WHERE q.cluster_id IS NOT NULL
        RETURN q.cluster_id AS cluster_id, q.text AS text
        ORDER BY q.cluster_id
    """)
    clusters = defaultdict(list)
    for record in result:
        clusters[record["cluster_id"]].append(record["text"])
    return clusters

# Fetch clusters from Neo4j
with driver.session() as session:
    print("📥 Fetching clusters from Neo4j...")
    clusters = session.read_transaction(fetch_clusters)
    print(f"📦 Found {len(clusters)} clusters.")


# Generate and write merged questions
print("🛠️ Generating merged questions...")
with open(OUTPUT_FILE, "w") as f:
    for cluster_id, questions in tqdm(clusters.items()):
        f.write(f"Cluster {cluster_id}:\n")
        for i, q in enumerate(questions, 1):
            f.write(f"  {i}. {q}\n")
            cur.execute(
             "INSERT INTO og_questions (classid,clusterid,question, answer) VALUES (%s, %s, %s, %s)",
             (0,int(cluster_id), q, "")
            )
            conn.commit()


        if len(questions) == 1:
            merged_question = questions[0]
        else:
            #prompt = "You are given multiple related questions. Combine them into one single, comprehensive question that captures all the important meanings and intentions of the original questions. : " + " ".join(questions)
            prompt = "Given the following list of questions about operating systems, write one comprehensive question that captures all their core ideas without omitting any important concept : " + " ".join(questions)
          
            response = generator(
				    prompt,
    				    max_length=128,
    				    do_sample=True,
    				    top_k=50,
    				    top_p=0.95,
    				    temperature=0.9,
   				    num_return_sequences=1
				)

            merged_question = response[0]["generated_text"].strip()
        cur.execute(
          "INSERT INTO questions (classid,clusterid,question, answer) VALUES (%s, %s, %s, %s)",
          (0,int(cluster_id), merged_question, "")
          )
        conn.commit()

        f.write(f"✅ Merged Question: {merged_question}\n\n")

print(f"✅ Written to {OUTPUT_FILE}")
cur.close()
conn.close()

