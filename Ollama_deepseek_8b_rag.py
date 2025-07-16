from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, helpers
from pathlib import Path
import requests, re, time, sys, os

# ─── CONFIG ──────────────────────────────────────────────────────────
HOST, PORT      = "localhost", 9200
INDEX_NAME      = "medical_pages"
TXT_PATH        = r"C:/Users/jahna/Documents/Demo-RAG/Hyponatremia Synthetic Pos Complex.txt"

EMBED_MODEL     = "intfloat/e5-base-v2"

# --- Ollama settings -------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = "deepseek-r1:8b"      # change/tag at will
OLLAMA_TIMEOUT  = 300
# --------------------------------------------------------------------
TOP_K           = 1
# ─────────────────────────────────────────────────────────────────────

embedder = SentenceTransformer(EMBED_MODEL)
client   = OpenSearch([{"host": HOST, "port": PORT}], use_ssl=False)

# ─── Wait for OpenSearch ────────────────────────────────────────────
print("⏳ Waiting for OpenSearch …")
for _ in range(30):
    if client.ping(): break
    time.sleep(2)
else:
    sys.exit("❌ OpenSearch not reachable on http://localhost:9200")

print("✅ OpenSearch is live")

# ─── Wait for Ollama (optional but helpful) ─────────────────────────
print("⏳ Checking Ollama …")
for _ in range(15):
    try:
        tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if tags.status_code == 200:
            print("✅ Ollama is up")
            break
    except requests.exceptions.RequestException:
        pass
    time.sleep(2)
else:
    print("⚠️  Could not reach Ollama; assuming it will start soon")

# ─── Ollama chat helper (chat → generate fallback) ──────────────────
def ollama_chat(prompt: str,
                temperature: float = 0.3,
                max_tokens: int = 256) -> str:
    # Prefer /api/chat
    payload_chat = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "options": {"num_predict": max_tokens},
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat",
                          json=payload_chat, timeout=OLLAMA_TIMEOUT)
        if r.status_code != 404:        # chat endpoint exists
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
    except requests.exceptions.JSONDecodeError:
        pass  # fall through to /api/generate

    # Fallback: /api/generate (older Ollama builds)
    payload_gen = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                      json=payload_gen, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    return r.json()["response"].strip()

# ─── Index helpers ──────────────────────────────────────────────────
def load_pages(fp):
    txt = Path(fp).read_text(encoding="utf-8")
    pat = r"\[PAGE (\d+) START\](.*?)\[PAGE \1 END\]"
    return [(int(n), c.strip()) for n, c in re.findall(pat, txt, re.DOTALL)]

def create_index():
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    mapping = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "page": {"type": "integer"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene"
                    } } } } }
    client.indices.create(index=INDEX_NAME, body=mapping)

def index_pages(pages):
    bulk = [{"_index": INDEX_NAME,
             "_source": {
                 "page": p,
                 "text": t,
                 "embedding": embedder.encode(f"passage: {t}").tolist()
             }} for p, t in pages]
    helpers.bulk(client, bulk)
    client.indices.refresh(index=INDEX_NAME)
    print(f"✅ Indexed {len(pages)} pages")

def retrieve(q, k=TOP_K):
    qvec = embedder.encode(f"query: {q}").tolist()
    body = {"size": k,
            "query": {"knn": {"embedding": {"vector": qvec, "k": k}}}}
    hits = client.search(index=INDEX_NAME, body=body)["hits"]["hits"]
    return [{"page": h["_source"]["page"],
             "score": h["_score"],
             "text": h["_source"]["text"]} for h in hits]

# ─── RAG answer step ────────────────────────────────────────────────
def answer(question, hits):
    if not hits:
        print("⚠️  No context found!\n"); return
    ctx = "\n\n---\n\n".join([f"[PAGE {h['page']}]\n{h['text']}" for h in hits])
    print("\n📚 CONTEXT PASSED TO LLM:\n" + "-"*60 + f"\n{ctx}\n" + "-"*60)
    prompt = (
        "You are a clinical coding assistant.\n"
        "Use ONLY the information inside <context> to answer the question.\n"
        "Answer in ONE short sentence. Do NOT reveal your reasoning.\n\n"
        f"<context>\n{ctx}\n</context>\n\n"
        f"Question: {question}\nAnswer:"
    )

    answer_text = ollama_chat(prompt, temperature=0.1, max_tokens=128)
    print("\n🧠 FINAL ANSWER:\n" + answer_text + "\n")

# ─── Run demo ───────────────────────────────────────────────────────
if __name__ == "__main__":
    create_index()
    index_pages(load_pages(TXT_PATH))

    QUESTIONS = [
        "Did patient had rash?",
        "What is the reason patient admitted?"
    ]
    for q in QUESTIONS:
        print(f"\n🔎 QUESTION: {q}")
        hits = retrieve(q)
        for h in hits:
            print(f"   • Page {h['page']}, score {h['score']:.3f}")
        answer(q, hits)
