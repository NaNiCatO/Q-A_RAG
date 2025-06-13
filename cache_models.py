# cache_models.py
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# Define a specific cache folder inside the container.
# This makes it predictable where the models will be stored.
cache_dir = "/app/model_cache"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

print("--- Starting Model Caching ---")

# Trigger the download of the retrieval model
print("Caching retrieval model: 'paraphrase-multilingual-MiniLM-L12-v2'")
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Retrieval model cached successfully.")

# Trigger the download of the re-ranking model
print("Caching re-ranking model: 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'")
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
print("Re-ranking model cached successfully.")

print(f"--- All models have been downloaded and cached in {cache_dir} ---")