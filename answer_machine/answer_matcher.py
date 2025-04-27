import os
import psycopg2
from transformers import pipeline
from collections import defaultdict

# Load FLAN-Alpaca-Base model
generator = pipeline(
    "text2text-generation",
    model="declare-lab/flan-alpaca-base",
    tokenizer="declare-lab/flan-alpaca-base"
)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("PGHOST", "postgres"),
    database=os.getenv("PGDATABASE", "qa_db"),
    user=os.getenv("PGUSER", "user"),
    password=os.getenv("PGPASSWORD", "password"),
    port=os.getenv("PGPORT", "5432")
)
cursor = conn.cursor()

# Fetch cluster and original questions
cursor.execute("""
    SELECT q.clusterid, q.question, og.question, og.questionid
    FROM questions q
    JOIN og_questions og ON q.clusterid = og.clusterid
    ORDER BY q.clusterid, og.questionid
""")
data = cursor.fetchall()

# Group by clusterid
cluster_data = defaultdict(list)
for clusterid, agg_q, og_q, og_id in data:
    cluster_data[clusterid].append({"agg": agg_q, "og": og_q, "og_id": og_id})

# Answer merged questions and rewrite specific answers
output_lines = []
for clusterid, items in cluster_data.items():
    agg_question = items[0]["agg"]
    print(f"\n📦 Cluster {clusterid}:")
    print(f"🧠 Aggregate Question: {agg_question}")
    answer = input("💬 Your answer: ")

    output_lines.append(f"Cluster {clusterid}:\nAggregate Question: {agg_question}\nAnswer: {answer}\n")

    for item in items:
        og_question = item["og"]
        prompt = (
            f"You are given the following answer: \"{answer}\" to a general question: \"{agg_question}\".\n"
            f"Now rewrite the answer so that it precisely addresses this specific question: \"{og_question}\"."
        )
        response = generator(prompt, max_length=128, truncation=True)
        modified_answer = response[0]["generated_text"].strip()

        output_lines.append(f"Original Question: {og_question}")
        output_lines.append(f"Modified Answer: {modified_answer}\n")

# Save to file
with open("answered_and_modified_flanalpaca.txt", "w") as f:
    f.write("\n".join(output_lines))

print("✅ All responses saved to answered_and_modified_flanalpaca.txt")

