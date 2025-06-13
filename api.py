import redis
import numpy as np
import requests
import os
import json 
import re 
# UPDATED: Import CrossEncoder from sentence-transformers
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# --- 1. INITIALIZATION ---

app = FastAPI(
    title="Coding Standards RAG API",
    description="An API with Hybrid Search and a Re-ranking Model.",
    version="4.2.0", # Version bump for new reranker model
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Model Loading ---
print("Loading sentence-transformer model for retrieval...")
# This model is for turning the query into a vector (retrieval)
retrieval_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Retrieval model loaded.")

# UPDATED: Swapped the English-centric model for a powerful multilingual one.
print("Loading Multilingual Cross-Encoder model for re-ranking...")
# This model is specifically designed for multilingual relevance scoring.
rerank_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
print("Re-ranking model loaded.")


print("Connecting to Redis...")
try:
    # r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    # Connect to the redis-stack service defined in docker-compose.yml
    r = redis.Redis(host='redis-stack', port=6379, decode_responses=True)
    r.ping()
    print("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    exit()

INDEX_NAME = "coding_standards_idx"

# --- 2. DATA MODELS for API ---

class QueryRequest(BaseModel):
    question: str

class ContextChunk(BaseModel):
    id: str
    content: str

class RelevantRule(BaseModel):
    rule_id: str
    is_relevant: bool
    reasoning: str
    content: str

class StructuredAnswer(BaseModel):
    summary: str
    detailed_explanation: str
    relevant_rules: list[RelevantRule]

class QueryResponse(BaseModel):
    structured_answer: StructuredAnswer
    raw_context: list[ContextChunk]


# --- 3. HELPER FUNCTIONS ---

def self_implemented_ngrams(text: str, n: int) -> list[str]:
    """A self-implemented function to generate n-grams from a string."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def generate_search_tokens(text: str) -> str:
    """Generates a space-separated string of trigrams (n=3) for search indexing."""
    sanitized_text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text_no_spaces = re.sub(r'\s+', '', sanitized_text)
    if not text_no_spaces: return ""
    tokens = self_implemented_ngrams(text_no_spaces, n=3)
    return " ".join(tokens)

def detect_language(text: str) -> str:
    """Detects if the text contains any Thai characters."""
    if re.search(r'[\u0E00-\u0E7F]', text):
        return "Thai"
    return "English"

def parse_redis_results(raw_results):
    """A helper to parse the flat list response from a raw FT.SEARCH command."""
    count = raw_results[0]
    docs = []
    for i in range(1, len(raw_results), 2):
        doc_data = {'doc_key': raw_results[i]}
        fields = raw_results[i+1]
        for j in range(0, len(fields), 2):
            doc_data[fields[j]] = fields[j+1]
        docs.append(doc_data)
    return count, docs

def hybrid_search(question: str, question_embedding: np.ndarray, lang: str) -> list[dict]:
    """Performs a hybrid search to retrieve an expanded set of candidate documents."""
    
    # Keyword Search Query
    keyword_query_str = ""
    if lang == "Thai":
        field = "content_th_display"
        thai_tokens = generate_search_tokens(question).split()
        if thai_tokens: keyword_query_str = f"@content_th_search:({'|'.join(thai_tokens)})"
    else: # English
        field = "content_en"
        sanitized_query = re.sub(r'[^\w\s]', ' ', question).strip()
        if sanitized_query:
            eng_tokens = sanitized_query.split()
            wildcard_tokens = [f"*{token}*" for token in eng_tokens]
            keyword_query_str = f"@content_en:({'|'.join(wildcard_tokens)})"

    print(f"\n--- Executing Hybrid Search (Phase 1: Retrieval) ---")
    
    # Execute Keyword Search
    keyword_docs = []
    if keyword_query_str:
        print(f"  - Keyword Query: FT.SEARCH {INDEX_NAME} \"{keyword_query_str}\"")
        command_args = ["FT.SEARCH", INDEX_NAME, keyword_query_str, "RETURN", 1, "ref_id"]
        try:
            raw_keyword_results = r.execute_command(*command_args)
            print(f"  - Raw Keyword Results: {raw_keyword_results}")
            count, keyword_docs = parse_redis_results(raw_keyword_results)
        except Exception as e: print(f"  - Keyword search failed: {e}")

    # Vector Search Query - Fetch more results (e.g., 10) for the re-ranker
    from redis.commands.search.query import Query
    vector_query = (
        Query("*=>[KNN 10 @embedding $query_vec as vector_score]")
        .sort_by("vector_score")
        .return_fields("ref_id", field)
        .dialect(2)
    )
    query_params = {"query_vec": question_embedding.astype(np.float32).tobytes()}
    vector_results = r.ft(INDEX_NAME).search(vector_query, query_params)
    
    print(f"  - Found {len(keyword_docs)} keyword results and {len(vector_results.docs)} vector results.")

    # Combine and de-duplicate results
    combined_results = {}
    for doc in keyword_docs:
        doc_content = r.hget(doc['doc_key'], field)
        combined_results[doc['ref_id']] = {"id": doc['ref_id'], "content": doc_content}

    for doc in vector_results.docs:
        if doc.ref_id not in combined_results:
            combined_results[doc.ref_id] = {"id": doc.ref_id, "content": getattr(doc, field)}

            
    final_chunks = list(combined_results.values())
    print(f"  - Combined to {len(final_chunks)} unique candidate documents.")
    return final_chunks

def rerank_documents(question: str, documents: list[dict]) -> list[dict]:
    """Re-ranks documents based on their relevance to the question using a Cross-Encoder."""
    if not documents:
        return []

    print(f"\n--- Executing Re-ranking (Phase 2: Scoring) ---")
    pairs = [(question, doc['content']) for doc in documents]
    scores = rerank_model.predict(pairs)
    
    for i in range(len(documents)):
        documents[i]['rerank_score'] = scores[i]
        
    sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
    
    print("  - Re-ranking complete. Scores:")
    for doc in sorted_docs:
        print(f"    - ID: {doc['id']}, Score: {doc['rerank_score']:.4f}")
        
    return sorted_docs[:8]

def generate_structured_llm_response(question: str, detected_language: str, context: list[dict]) -> dict:
    """Generates a structured JSON response from the Gemini API."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Google AI API key is not configured.")

    if not context:
        return {
            "summary": "I could not find any relevant coding standards for your question.",
            "detailed_explanation": "Please try rephrasing your query or check if the knowledge base contains information on this topic.",
            "relevant_rules": []
        }

    context_for_prompt = "\n".join([f"Rule ID: {chunk['id']}\nContent: {chunk['content']}\n---" for chunk in context])

    # UPDATED PROMPT: Added an explicit instruction to evaluate every single rule provided.
    prompt = f"""
        You are a smart code standards assistant. You will be given a user's question and a list of potentially relevant coding standard rules.

        Your task is to analyze every single rule and provide a structured JSON output. Follow these steps precisely:
        1.  Analyze the user's QUESTION.
        2.  For EACH of the PROVIDED RULES, you must make a determination. The 'relevant_rules' array in your final JSON output MUST contain one object for EACH of the PROVIDED RULES.
        3.  After evaluating all rules, construct a final 'summary' and 'detailed_explanation' using ONLY the rules you marked as relevant (is_relevant: true). If no rules are relevant, state that clearly in the summary and explanation.
        4.  Format your entire output as a single, valid JSON object with the specified structure.

            {{
              "summary": "A concise, one-sentence answer. This MUST be in the RESPONSE_LANGUAGE specified below.",
              "detailed_explanation": "A longer, paragraph-style explanation. This MUST be in the RESPONSE_LANGUAGE specified below.",
              "relevant_rules": [
                {{
                  "rule_id": "The ID of the rule you are evaluating.",
                  "is_relevant": "A boolean (true or false).",
                  "reasoning": "A short sentence explaining WHY the rule is or is not relevant to the user's question."
                }}
              ]
            }}

        RESPONSE_LANGUAGE: "{detected_language}"
        PROVIDED RULES:
        ---
        {context_for_prompt}
        ---
        USER'S QUESTION: "{question}"

        YOUR JSON OUTPUT:
    """
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        response = requests.post(api_url, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        
        raw_json_text = result["candidates"][0]["content"]["parts"][0]["text"]
        clean_json_text = raw_json_text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json_text)
    except Exception as e:
        print(f"Error during LLM call or JSON parsing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the AI response.")


# --- 4. API ENDPOINT (UPDATED WORKFLOW) ---

@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    lang = detect_language(request.question)
    
    question_embedding = retrieval_model.encode(request.question)
    
    candidate_chunks = hybrid_search(request.question, question_embedding, lang)
    
    reranked_chunks = rerank_documents(request.question, candidate_chunks)
    
    structured_answer_dict = generate_structured_llm_response(request.question, lang, reranked_chunks)

    # Add full content to the response for display
    chunk_map = {chunk['id']: chunk['content'] for chunk in reranked_chunks}
    if "relevant_rules" in structured_answer_dict:
        for rule in structured_answer_dict["relevant_rules"]:
            rule['content'] = chunk_map.get(rule['rule_id'], "Content not found.")
    
    return QueryResponse(
        structured_answer=structured_answer_dict, 
        raw_context=reranked_chunks
    )
