import asyncio
import inspect
import re
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any

import aiofiles
import faiss
import numpy as np
import orjson
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from rich.console import Console

import torch,logging
from rich.table import Table
from sentence_transformers import SentenceTransformer, CrossEncoder



_cpu_executor = ThreadPoolExecutor(max_workers=4)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger("SniperRAG")

def vanilla_parser(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Standard parser: Extracts primary text and preserves metadata.
    Handles common keys like 'text', 'body', or 'content'.
    """
    # Identify the primary text content
    content = doc.get("content") or doc.get("text") or doc.get("body") or ""

    # Clean up the doc to create metadata (everything except the main text)
    metadata = {k: v for k, v in doc.items() if k not in ["content", "text", "body"]}

    # Return as a list of rows (allows for future chunking/splitting)
    return [{
        "content": str(content),
        "metadata": metadata
    }]


# Mapping dictionary for dynamic lookup
PARSING_STRATEGIES = {
    "vanilla": vanilla_parser,
}
import polars as pl
from typing import Tuple
import google.genai as genai
class SourceDocument(BaseModel):
    rank: int
    score: float
    snippet: str
    metadata: dict = Field(default_factory=dict)
class RAGResponse(BaseModel):
    answer: str
    sources: List
    query: str

class GeminiGenerator:

    def __init__(self, api_key: str):

        self.client = genai.Client(api_key=api_key)

        # Standard RAG System Prompt
        self.system_instruction = (
            "You are a helpful AI assistant. Answer the user's question ONLY using the provided context. "
            "If the answer is not in the context, say you don't know. "
            "Always cite the 'Source Rank' when mentioning information."
        )

    async def generate_answer(self, query: str, search_results: List[Dict[str, Any]]) -> RAGResponse:
        """
        Combines retrieved context into a prompt and calls Gemini asynchronously.
        """
        # Format the context for the LLM
        context_blocks = []
        sources = []

        for res in search_results:
            context_blocks.append(f"Source Rank {res['rank']}:\n{res['snippet']}")
            sources.append(SourceDocument(
                rank=res['rank'],
                score=res['rerank_score'],
                snippet=res['snippet'],
                metadata=res['doc']
            ))

        context_str = "\n\n".join(context_blocks)

        full_prompt = (
            f"{self.system_instruction}\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"USER QUERY: {query}"
        )

        try:
            # 1. Call the method without 'await' first to capture the object
            res_or_coro = self.client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=full_prompt
            )

            # 2. BETTER SOLN: Check if it's a coroutine before awaiting
            if inspect.iscoroutine(res_or_coro) or asyncio.iscoroutine(res_or_coro):
                response = await res_or_coro
            else:
                # It's already the result object (GenerateContentResponse)
                response = res_or_coro

            answer_text = response.text
        except Exception as e:
            answer_text = f"Error generating answer: {str(e)}"

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            query=query
        )


class PerformanceSniperGateway:

    def __init__(self, searcher, generator, cache_path: str = "sniper_cache.parquet"):
        self.searcher : AsyncRAGSearcher = searcher
        self.generator : GeminiGenerator= generator
        self.cache_path = cache_path

        # Threshold Tuning
        self.MATCH_THRESHOLD = 0.95  # > 95%: Exact same intent
        self.DIFF_THRESHOLD = 0.82  # 82-95%: Follow-up or slight variation

        try:
            self.history = pl.read_parquet(cache_path)
        except FileNotFoundError:
            self.history = pl.DataFrame(schema={
                "query": pl.String, "response": pl.String, "embedding": pl.List(pl.Float32)
            })

    async def process_query(self, query: str) -> RAGResponse:
        # 1. Calculate Similarity once
        query_emb =  self.searcher.bi_encoder.encode([query])
        best_match_text, similarity = self._find_best_cache_hit(query, query_emb)

        # --- TIER 1: ZERO-LLM CACHE HIT ($0 / ~10ms) ---
        if similarity >= self.MATCH_THRESHOLD:
            # We return IMMEDIATELY. No LLM is called.
            logging.info("%s falled under cache hit",query)
            return RAGResponse(answer=best_match_text, sources=[], query=query)

        # --- TIER 2: THE DIFF HACK (Low Token / ~500ms) ---
        elif self.DIFF_THRESHOLD <= similarity < self.MATCH_THRESHOLD:
            # Only one skinny LLM call for the 'delta'.
            logging.info("%s falled under diff hack", query)
            #return await self._generate_diff(query, last_response=best_match_text)

        # --- TIER 3: FULL RAG PIPELINE (High Token / ~2s) ---
        # Only reached if Tier 1 and Tier 2 fail.
        logging.info("%s falled under full pipeline", query)
        results = await self.searcher.search(query)
        response = await self.generator.generate_answer(query, results)

        # Auto-learn: Save the expensive answer for future Tier 1 hits.
        if not response.answer.startswith("Error"):
            await self._update_cache(query=query,response=response.text,emb=query_emb)
        return RAGResponse(answer=response.answer, sources=[], query=query)
    @staticmethod
    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)


    async def _update_cache(self, query: str, response: str, emb: np.ndarray):
        """
        Appends a new query-response pair to the Polars history and persists it to disk.
        """
        # 1. Create a new row (using a list to ensure Polars treats it as a single row)
        new_row = pl.DataFrame({
            "query": [query],
            "response": [response],
            "embedding": [emb.tolist()]  # Ensure the embedding is a list for the Parquet column
        })

        # 2. Update the in-memory history
        if self.history.height == 0:
            self.history = new_row
        else:
            self.history = pl.concat([self.history, new_row])

        # 3. Persist to disk (Atomic write)
        # Using 'overwrite' logic; for very large caches, you might use 'sink_parquet'
        # but write_parquet is fine for a sniper cache.
        self.history.write_parquet(self.cache_path)
    async def _generate_diff(self, query: str, last_response: str) -> RAGResponse:
        """
        Token-Saving Strategy: Instead of sending whole documents,
        we send the previous answer as context.
        """
        diff_prompt = (
            f"PREVIOUS CONTEXT: {last_response}\n\n"
            f"NEW INSTRUCTION/QUERY: {query}\n\n"
            "TASK: Based on the previous context, provide the updated information. "
            "Only provide the delta (the changes) or the specific answer requested. "
            "Be extremely concise."
        )
        # Using the existing client from your GeminiGenerator
        res_or_coro =  self.generator.client.aio.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=diff_prompt
        )
        if inspect.iscoroutine(res_or_coro) or asyncio.iscoroutine(res_or_coro):
            response = await res_or_coro
        else:
            # It's already the result object (GenerateContentResponse)
            response = res_or_coro
        return RAGResponse(answer=response.text, sources=[], query=query)

    def _find_best_cache_hit(self, query: str, query_emb: np.ndarray) -> Tuple[str, float]:
        if self.history.height == 0:
            return "", 0.0

        # 1. LAYER ONE: Exact String Match (The "Short-circuit")
        # Using Polars to quickly filter for the exact string
        exact_match = self.history.filter(pl.col("query") == query)
        if not exact_match.is_empty():
            return exact_match[0, "response"], 1.0  # Return 1.0 as perfect similarity

        # 2. LAYER TWO: Semantic Match (Vector Search)
        # Convert list of embeddings back to a matrix
        embeddings = np.stack(self.history["embedding"].to_list())

        # Cosine similarity (assuming normalized embeddings)
        # query_emb[0] because it's usually (1, dim)
        embeddings = self.normalize(embeddings)
        query_emb = self.normalize(query_emb)
        scores = np.dot(embeddings, query_emb[0])
        best_idx = np.argmax(scores)
        return self.history[int(best_idx), "response"], float(scores[best_idx])
class VectorStore:
    def __init__(self):
        self.d = 0
        self.index = None
        self._lock = asyncio.Lock()
        self.min_ivf_size: int = 1000
        self.nlist: int = 50
    def build_sync(self, embeddings: np.ndarray,query_emb:np.ndarray):
        """Performs FAISS search, dynamically choosing between IVF and Flat index based on data size.
            Hardened: ensures dtype, shapes, contiguity and safe nlist.
            """

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array (N, d)")

        N, d = embeddings.shape

        # Ensure float32 and contiguous memory
        embeddings = np.asarray(embeddings, dtype=np.float32, order='C')
        query_emb = np.asarray(query_emb, dtype=np.float32, order='C')
        # Ensure query is 2D (nq, d)
        if query_emb.ndim == 1:
            if query_emb.shape[0] != d:
                raise ValueError(f"Query vector length ({query_emb.shape[0]}) doesn't match embedding dim ({d})")
            query_emb.reshape(1, -1)
        elif query_emb.ndim == 2 and query_emb.shape[1] != d:
            raise ValueError(f"Query second dimension ({query_emb.shape[1]}) doesn't match embedding dim ({d})")

        # Choose index strategy
        if N < self.min_ivf_size:
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)  # type: ignore
        else:
            safe_nlist = max(1, min(self.nlist, max(1, N // 10)))
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, safe_nlist, faiss.METRIC_L2)
            # Train only if needed
            if not index.is_trained:
                index.train(embeddings)
            index.add(embeddings)
        self.index = index
        return  index

    def add_incremental(self, embedding: np.ndarray):
        """Adds to index without full rebuild for small/medium chat scales."""
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        if self.index is None:
            self.d = emb.shape[1]
            self.index = faiss.IndexFlatL2(self.d)

        start = time.perf_counter()
        self.index.add(emb)
        logger.info(f"FAISS: Added vector. Total size: {self.index.ntotal} (Time: {time.perf_counter() - start:.4f}s)")

def encode_sync(model, texts, convert_to_numpy=True, show_progress_bar=False):
    """Encodes a list of texts into embeddings using the SentenceTransformer model."""
    return model.encode(texts, convert_to_numpy=convert_to_numpy, show_progress_bar=show_progress_bar)


def predict_sync(cross, pairs):
    """Predicts scores for query-text pairs using the CrossEncoder model."""
    return cross.predict(pairs)
store = VectorStore()
async def faiss_use(embeddings,query_emb: np.ndarray, k: int = 20):
    index = store.index
    if index is None:
        loop = asyncio.get_running_loop()
        index = await loop.run_in_executor(_cpu_executor, store.build_sync, embeddings, query_emb)
    # Perform search; query_emb is (nq, d)
    distances, indices = index.search(query_emb, k)  # type: ignore
    return distances, indices

class ModelProvider:
    _bi_encoder : SentenceTransformer | None = None
    _reranker : CrossEncoder| None  = None

    @classmethod
    def get_bi_encoder(cls):
        """Lazy-load the Bi-Encoder only when needed."""
        if cls._bi_encoder is None:
            from sentence_transformers import SentenceTransformer
            # Optimized for inference: Using CPU/GPU explicitly and half-precision if supported
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._bi_encoder = SentenceTransformer("./my_encoder_onnx", backend="onnx",local_files_only=True)
        return cls._bi_encoder

    @classmethod
    def get_reranker(cls):
        """Lazy-load the Cross-Encoder."""
        if cls._reranker is None:
            from sentence_transformers import CrossEncoder
            #device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._reranker = CrossEncoder(model_name_or_path="./my_cross_encoder_onnx",backend="onnx",local_files_only=True)
        return cls._reranker

    @classmethod
    def clear_vram(cls):
        """Force manual cleanup if the system is under pressure."""
        cls._bi_encoder = None
        cls._reranker = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
class AsyncRAGSearcher:
    """An asynchronous RAG searcher class handling corpus loading, embedding, and two-stage retrieval."""

    def __init__(self, corpus_path: str,
                 corpus_format: str,
                 executor_workers: Optional[int] = None):
        self.corpus_path = corpus_path
        self.corpus_format = corpus_format
        self.console = Console()
        self.corpus: List[Dict[str, Any]] = []
        self.processed: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.schema_report: Dict[str, Any] = {}

        self.parser_func = PARSING_STRATEGIES[self.corpus_format]

        # models and executor
        self.provider = ModelProvider()
        self.bi_encoder = ModelProvider.get_bi_encoder()
        self.cross_encoder = ModelProvider.get_reranker()
        self.executor = ThreadPoolExecutor(max_workers=executor_workers)

    # -------------------------
    # Loading + schema inference
    # -------------------------
    async def load_corpus(self):
        """Asynchronously loads and parses the JSON corpus file."""
        try:
            async with aiofiles.open(self.corpus_path, "rb") as f:
                data = orjson.loads(await f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        except orjson.JSONDecodeError:
            raise ValueError(f"Could not decode JSON from file: {self.corpus_path}")

        if not isinstance(data, list):
            raise ValueError("Expected top-level JSON list of documents.")

        all_rows = []
        # DYNAMIC CALL: Use the selected parser function
        for doc in data:
            all_rows.extend(self.parser_func(doc))

        if not all_rows:
            raise ValueError(
                f"Corpus is empty after processing with format '{self.corpus_format}'! "
                "Check data structure or switch to a different format key."
            )

        self.corpus = all_rows
        self.processed = []
        for i, row in enumerate(all_rows):
            if isinstance(row, dict) and "content" in row:
                self.processed.append(row["content"])
            else:
                # This will tell you EXACTLY what is breaking your RAG
                print(f"⚠️ ERROR: Row {i} is invalid! Data: {row}")
        self.schema_report = self.build_schema_report(sample_n=5)

    def build_schema_report(self, sample_n: int = 3) -> Dict[str, Any]:
        """Returns a small report describing keys / shapes found in the corpus."""
        report = {"total_documents": len(self.corpus), "top_level_keys": {}, "nested_examples": {}}
        for i, doc in enumerate(self.corpus[:sample_n]):
            for k, v in doc.items():
                t = type(v).__name__
                report["top_level_keys"].setdefault(k, set()).add(t)
                if isinstance(v, list) and v:
                    first = v[0]
                    if isinstance(first, dict):
                        report["nested_examples"].setdefault(k, set()).update(first.keys())

        report["top_level_keys"] = {k: list(v) for k, v in report["top_level_keys"].items()}
        report["nested_examples"] = {k: list(v) for k, v in report["nested_examples"].items()}
        return report

    def get_schema_report(self) -> Dict[str, Any]:
        """Retrieves the corpus schema report."""
        return self.schema_report

    # -------------------------
    # Encode corpus (async-safe)
    # -------------------------
    async def encode_corpus(self, show_progress: bool = True):
        """Encodes the entire corpus into embeddings asynchronously."""
        if not self.processed:
            raise ValueError("Corpus empty — load_corpus() first.")
        loop = asyncio.get_running_loop()
        self.corpus_embeddings = await loop.run_in_executor(
            self.executor,
            encode_sync,
            self.bi_encoder,
            self.processed,
            True,
            show_progress
        )

    @staticmethod
    async def clean_snippet(txt: str, min_chunk_len: int = 30, similarity_thresh: float = 0.95) -> str:
        """Cleans a large text snippet by normalizing, chunking, and deduplicating."""
        if not txt or not txt.strip():
            return ""
        # 1️⃣ Normalize spaces and remove excessive newlines
        txt = re.sub(r'\s+', ' ', txt.strip())
        # 2️⃣ Split by periods, double spaces, or newlines
        chunks = re.split(r'\. |\s{2,}|\n{2,}', txt)
        # 3️⃣ Deduplicate using exact and fuzzy matching
        unique_chunks = []

        def is_duplicate(new_chunk):
            for uc in unique_chunks:
                ratio = SequenceMatcher(None, uc.lower(), new_chunk.lower()).ratio()
                if ratio > similarity_thresh:
                    return True
            return False

        for chunk in chunks:
            chunk_clean = chunk.strip()
            if len(chunk_clean) < min_chunk_len:
                continue
            if not is_duplicate(chunk_clean):
                unique_chunks.append(chunk_clean)
        # 4️⃣ Join the cleaned chunks
        return '. '.join(unique_chunks) + '.'

    # -------------------------
    # Search
    # -------------------------
    async def search(self,
                     query: str,
                     top_k: int = 20,
                     rerank_top: int = 5,
                     filter_key: Optional[str] = None,
                     filter_value: Optional[Any] = None,print_result=True) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        if filter_key is not None:
            indices = [i for i, d in enumerate(self.corpus) if d.get(filter_key) == filter_value]
            if not indices:
                return []
            processed_subset = [self.processed[i] for i in indices]
        else:
            indices = list(range(len(self.corpus)))
            processed_subset = self.processed

            # Get embeddings for the subset
        if self.corpus_embeddings is not None and len(indices) == len(self.corpus):
            emb_subset = self.corpus_embeddings
        elif self.corpus_embeddings is not None and len(indices) != len(self.corpus):
            emb_subset = self.corpus_embeddings[np.array(indices)]
        else:
            emb_subset = await loop.run_in_executor(
                self.executor,
                encode_sync,
                self.bi_encoder,
                processed_subset,
                True,
                False
            )
            # encode query
        query_emb_list = await loop.run_in_executor(self.executor, encode_sync, self.bi_encoder, [query], True, False)
        query_emb = query_emb_list[0]
        query_emb = np.array(query_emb)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
            # ... FAISS SEARCH ...
        distances_faiss, indices_faiss = await faiss_use(embeddings=emb_subset,query_emb=query_emb,k=top_k)

        # SAFETY CHECK 1: Ensure FAISS returned results
        if len(indices_faiss) == 0 or len(indices_faiss[0]) == 0:
                print("⚠️ FAISS returned no results.")
                return []

        candidates = []
        for cid, distance in zip(indices_faiss[0], distances_faiss[0]):
            if cid == -1: continue  # FAISS returns -1 if not enough neighbors found
            orig_idx = indices[cid]
            candidates.append((processed_subset[cid], self.corpus[orig_idx], orig_idx, float(distance)))

            # SAFETY CHECK 2: Robust BM25 Mapping
        texts_for_bm25 = [cand[0].split() for cand in candidates]
        bm25 = BM25Okapi(texts_for_bm25)

            # Use index-based mapping instead of string comparison
        doc_scores = bm25.get_scores(query.split())
            # Sort candidates by BM25 score
        ranked_indices = np.argsort(doc_scores)[::-1]
        candidates = [candidates[i] for i in ranked_indices]
        # ----------------------------------------------------------------------
        # RERANKING STAGE
        # ----------------------------------------------------------------------
        pairs = [(query, cand[0]) for cand in candidates]

        if not pairs:
            return []

        scores = await loop.run_in_executor(self.executor, predict_sync, self.cross_encoder, pairs)

        # rerank by cross-encoder score (highest score is best)
        reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        # Prepare return results
        results = []
        keywords = re.findall(r'\w+', query.lower())
        table = Table(
            title=f"Top results for query: '[bold yellow]{query}[/bold yellow]' (Format: {self.corpus_format})")
        table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
        table.add_column("Score (rerank)", justify="right", style="green")
        table.add_column("Snippet", justify="left", style="white")

        for rank, (score, (flat_text, raw_doc, orig_idx, embed_score)) in enumerate(reranked[:rerank_top], start=1):
            snippet = flat_text.replace("\n", " ")
            highlighted = re.sub(
                rf'(\b({"|".join(re.escape(k) for k in keywords)})\b)',
                r'[bold yellow]\1[/bold yellow]',
                snippet,
                flags=re.IGNORECASE
            ) if keywords else snippet
            table.add_row(str(rank), f"{float(score):.4f}", highlighted)
            results.append({
                "rank": rank,
                "rerank_score": float(score),
                "embed_score": float(embed_score),  # This is the FAISS distance (lower is better)
                "doc_index": int(orig_idx),
                "snippet": await self.clean_snippet(snippet),
                "doc": raw_doc
            })
        if print_result :
            self.console.print(table)
        return results

    # Add this method inside your AsyncRAGSearcher class
    async def add_to_index(self, text: str, metadata: dict = None):
        """Adds a new text snippet to the live corpus and updates embeddings."""
        if not text or text.strip() == "":
            return

        # 1. Clean and wrap the snippet
        clean_text = await self.clean_snippet(text)  #
        doc_entry = {"content": clean_text, "metadata": metadata or {}}

        # 2. Append to internal corpus lists
        self.corpus.append(doc_entry)
        self.processed.append(clean_text)

        # 3. Update embeddings (Incremental)
        loop = asyncio.get_running_loop()
        new_emb = await loop.run_in_executor(
            self.executor,
            encode_sync,
            self.bi_encoder,
            [clean_text],
            True,
            False
        )

        if self.corpus_embeddings is None:
            self.corpus_embeddings = new_emb
        else:
            self.corpus_embeddings = np.vstack([self.corpus_embeddings, new_emb])

        # 4. Force FAISS index rebuild on next search
        # In VectorStore, build_sync is called if index is None
        store.index = None


async def chat_session():
    # --- INITIALIZATION ---
    # Start with an empty searcher (no corpus_path needed for dynamic mode)
    searcher = AsyncRAGSearcher(corpus_path="", corpus_format="vanilla")
    generator = GeminiGenerator(api_key="AIzaSyCByI0UWIvdlY5wQYZ_uMIew8t9Y8-Q71A")

    # Gateway still handles the "Diff Hack" for efficiency
    gateway = PerformanceSniperGateway(searcher, generator)

    print("🧠 Dynamic Brain Initialized. Start talking!")

    while True:
        user_input = input("\n👤 User: ").strip()
        if user_input.lower() in ["exit", "quit"]: break

        # 1. Gateway Process (Retrieval + Generation)
        response = await gateway.process_query(user_input)
        print(f"\n🤖 Assistant: {response.answer}")

        # 2. Dynamic Learning Step
        # Ensure we wait for the index to update so it's ready for the next turn
        knowledge_snippet = f"User: {user_input} | Assistant: {response.answer}"
        await searcher.add_to_index(knowledge_snippet, metadata={"type": "history"})

        # 3. Log Source Info
        if response.sources:
            print(f"🔗 [Found {len(response.sources)} relevant context blocks]")
