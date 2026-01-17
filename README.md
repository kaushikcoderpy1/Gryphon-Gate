# Gryphon-Gate
is a high-performance, asynchronous RAG (Retrieval-Augmented Generation) framework designed for speed, cost-efficiency, and dynamic learning. It features a unique **3-Tiered Gateway** that optimizes LLM usage by bypassing heavy computation for known or similar queries.

---

## 🚀 Key Features

* **3-Tiered Performance Gateway**:
* **Tier 1: Semantic Cache (Zero-LLM)**: Blazing fast (~10ms) responses for exact or near-identical queries using a Polars-backed Parquet cache.
* **Tier 2: The Diff Hack (Token Saver)**: Handles follow-up questions by using the previous answer as context instead of re-searching the entire database.
* **Tier 3: Full RAG Pipeline**: A heavy-duty retrieval process involving FAISS vector search, BM25 keyword matching, and Cross-Encoder reranking.


* **Dynamic Knowledge Injection**: The system learns in real-time. Every interaction is indexed and becomes part of the searchable corpus instantly.
* **Asynchronous & Optimized**: Built with `asyncio`, `aiofiles`, and `ThreadPoolExecutor`. Uses ONNX-exported models for sub-millisecond embedding and reranking latency.
* **Gemini Integration**: Built-in support for Google Gemini models with a robust "Dynamic Awaiter" to handle asynchronous SDK calls safely.

---

## 🛠️ Architecture

1. **Bi-Encoder**: Generates dense vector embeddings for retrieval.
2. **FAISS Vector Store**: Performs efficient similarity search across thousands of documents.
3. **Cross-Encoder**: Reranks the top candidates to ensure the most relevant context is sent to the LLM.
4. **Semantic Gateway**: Logic engine that decides if a query needs a full search or can be served from the cache.

---

## 📦 Installation

```bash
pip install asyncio aiofiles faiss-cpu numpy orjson pydantic rank_bm25 rich torch sentence_transformers polars google-genai

```

*Note: Ensure you have your ONNX-exported models in the `./my_encoder_onnx` and `./my_cross_encoder_onnx` directories.*

---

## 🚦 Quick Start

### 1. Initialize the Components

```python
from sniper_rag import AsyncRAGSearcher, GeminiGenerator, PerformanceSniperGateway

# Initialize Searcher (Retriever)
searcher = AsyncRAGSearcher(corpus_path="data.json", corpus_format="vanilla")

# Initialize Generator (Gemini)
generator = GeminiGenerator(api_key="YOUR_GEMINI_API_KEY")

# Initialize Gateway (Controller)
gateway = PerformanceSniperGateway(searcher, generator)

```

### 2. Run the Dynamic Chat Loop

SniperRAG is designed to "learn" as you talk to it.

```python
async def main():
    while True:
        user_input = input("👤 User: ")
        
        # Gateway handles Tier 1, 2, or 3 automatically
        response = await gateway.process_query(user_input)
        
        print(f"🤖 Assistant: {response.answer}")
        
        # Inject the new interaction back into the live index
        knowledge = f"User: {user_input} | Assistant: {response.answer}"
        await searcher.add_to_index(knowledge)

if __name__ == "__main__":
    asyncio.run(main())

```

---

## ⚙️ Threshold Tuning

The gateway behavior is controlled by two main thresholds:

* **`MATCH_THRESHOLD` (0.95)**: If query similarity is above this, the system returns the cached answer without calling the LLM.
* **`DIFF_THRESHOLD` (0.82)**: If similarity is in this range, the "Diff Hack" triggers, asking the LLM to provide only the "delta" or updated info based on the last response.

---

## 📊 Performance Benchmarks (Approximate)

| Tier | Latency | Token Cost | Accuracy |
| --- | --- | --- | --- |
| **Tier 1 (Cache)** | ~10-20ms | $0.00 | 100% (Identical Match) |
| **Tier 2 (Diff)** | ~400-600ms | Ultra-Low | High (Context-dependent) |
| **Tier 3 (Full RAG)** | ~1.5-2.5s | Standard | Maximum (Grounding) |

---

## 📝 License

MIT License. Built for developers who need RAG that doesn't just work, but scales and saves.
