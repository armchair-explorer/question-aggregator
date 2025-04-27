#!/bin/bash


echo "🧹 Cleaning Docker containers and volumes..."

#docker compose down --volumes --remove-orphans
#docker builder prune --all --force

#docker system prune -a --volumes -f
#docker builder prune -a -f

echo "🧼 Docker cleaned."

set -e  # Stop on error


# Step 0: Start Neo4j in background
echo "🚀 Starting Neo4j..."
docker compose up -d neo4j

docker compose up -d postgres

# Step 1: Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to be ready..."
until nc -z localhost 7687; do
  sleep 1
done
echo "✅ Neo4j is up!"

# Step 2: Run vectorizer
echo "🚀 Running vectorizer..."
docker compose up vectorizer
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "❌ Vectorizer failed with exit code $EXIT_CODE"
  echo "🧹 Shutting down Neo4j..."
  docker compose down
  exit $EXIT_CODE
fi

docker compose build question-augmenter

# Step 4: Run question augmenter
echo "🚀 Running augmenter with transformers..."
docker compose up question-augmenter

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "❌ question augmenter failed with exit code $EXIT_CODE"
  echo "🧹 Shutting down Neo4j..."
  docker compose down
  exit $EXIT_CODE
fi





# Step 5: Cleanup
echo "🧹 Shutting down Neo4j..."
docker compose down -v  neo4j
docker compose down vectorizer
docker compose down question-augmenter

