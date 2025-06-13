import redis
import numpy as np
import json 
import re # Import re for our new tokenizer
from sentence_transformers import SentenceTransformer

from redis.commands.search.field import (
    VectorField,
    TagField,
    TextField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)

# --- 1. TOKENIZER & DATA LOADING ---

def self_implemented_ngrams(text: str, n: int) -> list[str]:
    """A self-implemented function to generate n-grams from a string."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def generate_search_tokens(text: str) -> str:
    """
    Generates a space-separated string of trigrams (n=3) for search indexing.
    This version correctly handles Unicode and Thai tone marks.
    """
    # 1. Sanitize by removing only punctuation, keeping all letters/numbers.
    # This regex is Unicode-aware.
    sanitized_text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    # 2. Remove all whitespace to create a contiguous string for n-grams.
    text_no_spaces = re.sub(r'\s+', '', sanitized_text)
    
    if not text_no_spaces:
        return ""
        
    # 3. Generate trigrams.
    tokens = self_implemented_ngrams(text_no_spaces, n=3)
    return " ".join(tokens)

def load_knowledge_base():
    """Loads the coding standards from the new bilingual JSON file."""
    try:
        with open("knowledge_base_bilingual.json", "r", encoding="utf-8") as f:
            print("Successfully loaded knowledge_base_bilingual.json")
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: knowledge_base_bilingual.json not found. Please create it.")
        return []
    except json.JSONDecodeError:
        print("ERROR: Could not decode JSON from file. Please check its format.")
        return []

# --- 2. INITIALIZE MODEL AND REDIS CONNECTION ---
print("Loading sentence-transformer model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded.")

try:
    # r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    # Connect to the redis-stack service defined in docker-compose.yml
    r = redis.Redis(host='redis-stack', port=6379, decode_responses=True)
    r.ping()
    print("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    exit()

# --- 3. CREATE THE HYBRID SEARCH INDEX ---
INDEX_NAME = "coding_standards_idx"
DOC_PREFIX = "doc:"
VECTOR_DIMENSION = 384 

schema = (
    TagField("ref_id", as_name="ref_id"), 
    TextField("content_en", as_name="content_en"),
    TextField("content_th_display", as_name="content_th_display"),
    TextField("content_th_search", as_name="content_th_search"),
    VectorField(
        "embedding",
        "FLAT", {"TYPE": "FLOAT32", "DIM": VECTOR_DIMENSION, "DISTANCE_METRIC": "COSINE"},
        as_name="embedding",
    ),
)

print("Checking for existing index...")
try:
    r.ft(INDEX_NAME).dropindex(delete_documents=True)
    print("Dropped existing index to apply new schema.")
except redis.exceptions.ResponseError:
    print("No existing index to drop. Creating a new one.")

definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)
r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
print(f"Created new Hybrid Search index: {INDEX_NAME}")


# --- 4. EMBED AND INGEST THE DATA ---
print("\nStarting data ingestion into Redis...")

KNOWLEDGE_BASE = load_knowledge_base()

if not KNOWLEDGE_BASE:
    print("Aborting ingestion because knowledge base is empty.")
    exit()

pipeline = r.pipeline()
for i, doc in enumerate(KNOWLEDGE_BASE):
    embedding = model.encode(doc['content']).astype(np.float32).tobytes()
    key = f"{DOC_PREFIX}{i}"
    
    doc_data = {
        "ref_id": doc['ref_id'],
        "content_en": doc['content'],
    }
    
    if 'content_th' in doc and doc['content_th']:
        doc_data['content_th_display'] = doc['content_th']
        # Generate and add the searchable trigram tokens using the new function
        doc_data['content_th_search'] = generate_search_tokens(doc['content_th'])
    
    doc_data['embedding'] = embedding
    
    pipeline.hset(key, mapping=doc_data)
    print(f"  - Preparing document {i+1}/{len(KNOWLEDGE_BASE)} (ID: {doc['ref_id']})")

results = pipeline.execute()
print("Pipeline executed.")
print(f"\nSuccessfully ingested {len(KNOWLEDGE_BASE)} documents into Redis with the new hybrid schema.")
