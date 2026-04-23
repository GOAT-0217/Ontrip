import re
import math
import time
import sqlite3
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger

settings = get_settings()


class RetrievalStrategy(Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class QueryType(Enum):
    SEMANTIC = "semantic"
    EXACT = "exact"
    COMPOSITE = "composite"


@dataclass
class RetrievalResult:
    content: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "vector"


@dataclass
class OrchestratorConfig:
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    rerank_top_k: int = 5
    relevance_threshold: float = 0.3
    max_rewrite_attempts: int = 2
    enable_query_rewrite: bool = True
    enable_rerank: bool = True
    enable_hybrid: bool = True


_EXACT_MATCH_PATTERNS = [
    re.compile(r'[A-Z]{2}\d{3,4}', re.IGNORECASE),
    re.compile(r'\bID\s*[:：]?\s*\d+', re.IGNORECASE),
    re.compile(r'\b编号\s*[:：]?\s*\d+'),
    re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'),
    re.compile(r'\b(?:航班号|车次|班次)\s*[:：]?\s*\S+'),
]


def _classify_query(query: str) -> QueryType:
    for pattern in _EXACT_MATCH_PATTERNS:
        if pattern.search(query):
            has_semantic = any(
                kw in query
                for kw in ['推荐', '建议', '适合', '好玩', '好吃', '便宜', '豪华',
                           'recommend', 'suggest', 'best', 'cheap', 'luxury',
                           '附近', '周边', 'around', 'near']
            )
            if has_semantic:
                return QueryType.COMPOSITE
            return QueryType.EXACT
    return QueryType.SEMANTIC


def _select_strategy(query_type: QueryType, config: OrchestratorConfig) -> RetrievalStrategy:
    if not config.enable_hybrid:
        return RetrievalStrategy.VECTOR
    strategy_map = {
        QueryType.SEMANTIC: RetrievalStrategy.VECTOR,
        QueryType.EXACT: RetrievalStrategy.KEYWORD,
        QueryType.COMPOSITE: RetrievalStrategy.HYBRID,
    }
    return strategy_map.get(query_type, RetrievalStrategy.VECTOR)


async def rewrite_query(query: str, context: str = "") -> List[str]:
    if not settings.OPENAI_API_KEY:
        return [query]

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL if settings.OPENAI_BASE_URL else None,
            temperature=0.3,
            max_tokens=200,
        )

        system_prompt = (
            "你是一个查询改写专家。将用户的搜索查询改写为更适合向量检索的形式。\n"
            "规则：\n"
            "1. 保留原始查询的核心意图\n"
            "2. 补充同义词和相关术语\n"
            "3. 将口语化表达转为更正式的描述\n"
            "4. 每行输出一个改写查询，不要编号，不要解释\n"
            "5. 输出2-3个改写版本"
        )

        user_prompt = f"原始查询: {query}"
        if context:
            user_prompt += f"\n上下文: {context}"

        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        rewrites = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
        all_queries = [query] + rewrites
        seen = set()
        unique = []
        for q in all_queries:
            normalized = q.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(q)
        return unique[:4]

    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return [query]


def _bm25_score(query: str, document: str, k1: float = 1.5, b: float = 0.75) -> float:
    def tokenize(text: str) -> List[str]:
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', text.lower())
        result = []
        for token in tokens:
            if re.match(r'[\u4e00-\u9fff]', token):
                for i in range(len(token)):
                    result.append(token[i])
                    if i < len(token) - 1:
                        result.append(token[i:i + 2])
            else:
                result.append(token)
        return result

    query_terms = tokenize(query)
    doc_terms = tokenize(document)
    if not query_terms or not doc_terms:
        return 0.0

    doc_len = len(doc_terms)
    avg_doc_len = max(doc_len, 50)

    term_freq = {}
    for term in doc_terms:
        term_freq[term] = term_freq.get(term, 0) + 1

    score = 0.0
    for term in query_terms:
        tf = term_freq.get(term, 0)
        if tf == 0:
            continue
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        idf = math.log(1 + (100 - tf + 0.5) / (tf + 0.5))
        score += tf_norm * idf

    return score


def keyword_search(
    query: str,
    table_name: str,
    db_path: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            conn.close()
            return []

        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        column_names = [col[0] for col in cursor.description]
        conn.close()

        if not rows:
            return []

        results = []
        for row in rows:
            record = dict(zip(column_names, row))
            doc_text = " ".join(str(v) for v in record.values() if v is not None)
            score = _bm25_score(query, doc_text)
            if score > 0:
                results.append({"payload": record, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    except Exception as e:
        logger.warning(f"Keyword search failed for {table_name}: {e}")
        return []


def vector_search(
    query: str,
    vectordb,
    limit: int = 5,
) -> List[RetrievalResult]:
    if vectordb is None:
        return []

    try:
        search_results = vectordb.search(query, limit=limit)
        results = []
        for result in search_results:
            payload = result.payload or {}
            content = payload.get("content", "")
            score = getattr(result, 'score', 0.0)
            if score == 0.0 and hasattr(result, 'distance'):
                score = 1.0 - result.distance
            results.append(RetrievalResult(
                content=content,
                score=score,
                payload=payload,
                source="vector",
            ))
        return results
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []


def hybrid_search(
    query: str,
    vectordb,
    table_name: str,
    db_path: str,
    limit: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[RetrievalResult]:
    vector_limit = limit * 2
    keyword_limit = limit * 2

    vector_results = vector_search(query, vectordb, limit=vector_limit)
    keyword_results = keyword_search(query, table_name, db_path, limit=keyword_limit)

    if not vector_results and not keyword_results:
        return []
    if not vector_results:
        return [
            RetrievalResult(
                content=" ".join(str(v) for v in r["payload"].values() if v is not None),
                score=r["score"] * keyword_weight,
                payload=r["payload"],
                source="keyword",
            )
            for r in keyword_results[:limit]
        ]
    if not keyword_results:
        return vector_results[:limit]

    merged = {}
    for vr in vector_results:
        key = _make_dedup_key(vr.payload)
        if key not in merged:
            merged[key] = RetrievalResult(
                content=vr.content,
                score=vr.score * vector_weight,
                payload=vr.payload,
                source="vector",
            )
        else:
            merged[key].score += vr.score * vector_weight

    for kr in keyword_results:
        key = _make_dedup_key(kr["payload"])
        content = " ".join(str(v) for v in kr["payload"].values() if v is not None)
        if key not in merged:
            merged[key] = RetrievalResult(
                content=content,
                score=kr["score"] * keyword_weight,
                payload=kr["payload"],
                source="keyword",
            )
        else:
            merged[key].score += kr["score"] * keyword_weight
            if merged[key].source == "vector":
                merged[key].source = "hybrid"

    results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
    return results[:limit]


def _make_dedup_key(payload: Dict) -> str:
    id_val = payload.get("id", payload.get("name", ""))
    return f"{id_val}"


def rerank_results(
    query: str,
    results: List[RetrievalResult],
    top_k: int = 5,
) -> List[RetrievalResult]:
    if not results:
        return results

    try:
        reranked = []
        for result in results:
            doc_text = result.content or " ".join(str(v) for v in result.payload.values() if v is not None)
            cross_score = _cross_encoder_score(query, doc_text)
            combined_score = 0.4 * result.score + 0.6 * cross_score
            reranked.append(RetrievalResult(
                content=result.content,
                score=combined_score,
                payload=result.payload,
                source=result.source,
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Reranking failed, using original order: {e}")
        return results[:top_k]


def _cross_encoder_score(query: str, document: str) -> float:
    query_terms = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', query.lower()))
    doc_terms = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', document.lower()))

    if not query_terms or not doc_terms:
        return 0.0

    overlap = query_terms & doc_terms
    if not overlap:
        bigram_overlap = 0
        query_bigrams = set()
        for term in query_terms:
            chars = re.findall(r'[\u4e00-\u9fff]', term)
            for i in range(len(chars) - 1):
                query_bigrams.add(chars[i] + chars[i + 1])
        doc_bigrams = set()
        for term in doc_terms:
            chars = re.findall(r'[\u4e00-\u9fff]', term)
            for i in range(len(chars) - 1):
                doc_bigrams.add(chars[i] + chars[i + 1])
        if query_bigrams and doc_bigrams:
            bigram_overlap = len(query_bigrams & doc_bigrams) / len(query_bigrams)
        return bigram_overlap * 0.5

    precision = len(overlap) / len(query_terms)
    recall = len(overlap) / len(doc_terms) if doc_terms else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    position_bonus = 0.0
    doc_lower = document.lower()
    for i, term in enumerate(query_terms):
        pos = doc_lower.find(term)
        if pos >= 0:
            position_bonus += 1.0 / (1 + pos * 0.01)
    position_bonus = min(position_bonus / len(query_terms), 1.0)

    return 0.6 * f1 + 0.4 * position_bonus


def assess_relevance(query: str, results: List[RetrievalResult], threshold: float = 0.3) -> Tuple[bool, float]:
    if not results:
        return False, 0.0

    top_score = results[0].score if results else 0.0
    avg_score = sum(r.score for r in results[:3]) / min(len(results), 3)

    confidence = 0.6 * top_score + 0.4 * avg_score
    is_relevant = confidence >= threshold

    return is_relevant, confidence


class RetrievalOrchestrator:
    def __init__(
        self,
        vectordb=None,
        table_name: str = "",
        db_path: str = "",
        config: Optional[OrchestratorConfig] = None,
        formatter=None,
    ):
        self.vectordb = vectordb
        self.table_name = table_name
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self.config = config or OrchestratorConfig()
        self.formatter = formatter

    async def search(
        self,
        query: str,
        limit: int = 5,
        context: str = "",
    ) -> List[RetrievalResult]:
        query_type = _classify_query(query)
        strategy = _select_strategy(query_type, self.config)

        logger.info(
            f"🔍 RetrievalOrchestrator: query='{query[:50]}...', "
            f"type={query_type.value}, strategy={strategy.value}"
        )

        queries = [query]
        if self.config.enable_query_rewrite and query_type != QueryType.EXACT:
            queries = await rewrite_query(query, context)
            if len(queries) > 1:
                logger.info(f"📝 Query rewritten: {queries}")

        all_results = []
        for q in queries:
            results = self._execute_search(q, strategy, limit)
            all_results.extend(results)

        all_results = self._deduplicate(all_results)

        if self.config.enable_rerank and len(all_results) > 1:
            all_results = rerank_results(query, all_results, top_k=self.config.rerank_top_k)
            logger.info(f"🔄 Reranked {len(all_results)} results")

        is_relevant, confidence = assess_relevance(
            query, all_results, self.config.relevance_threshold
        )

        if not is_relevant and self.config.max_rewrite_attempts > 0:
            logger.info(f"⚠️ Low relevance ({confidence:.3f}), attempting retry with different strategy")
            retry_results = await self._retry_with_different_strategy(
                query, strategy, limit, context
            )
            if retry_results:
                _, retry_confidence = assess_relevance(
                    query, retry_results, self.config.relevance_threshold
                )
                if retry_confidence > confidence:
                    all_results = retry_results
                    logger.info(f"✅ Retry improved relevance: {retry_confidence:.3f}")

        final_results = all_results[:limit]
        logger.info(
            f"📊 Retrieval complete: {len(final_results)} results, "
            f"strategy={strategy.value}, confidence={confidence:.3f}"
        )
        return final_results

    def _execute_search(
        self,
        query: str,
        strategy: RetrievalStrategy,
        limit: int,
    ) -> List[RetrievalResult]:
        if strategy == RetrievalStrategy.VECTOR:
            return vector_search(query, self.vectordb, limit=limit)
        elif strategy == RetrievalStrategy.KEYWORD:
            kw_results = keyword_search(query, self.table_name, self.db_path, limit=limit)
            return [
                RetrievalResult(
                    content=" ".join(str(v) for v in r["payload"].values() if v is not None),
                    score=r["score"],
                    payload=r["payload"],
                    source="keyword",
                )
                for r in kw_results
            ]
        elif strategy == RetrievalStrategy.HYBRID:
            return hybrid_search(
                query, self.vectordb, self.table_name, self.db_path,
                limit=limit,
                vector_weight=self.config.vector_weight,
                keyword_weight=self.config.keyword_weight,
            )
        return []

    async def _retry_with_different_strategy(
        self,
        query: str,
        original_strategy: RetrievalStrategy,
        limit: int,
        context: str,
    ) -> List[RetrievalResult]:
        strategy_fallback = {
            RetrievalStrategy.VECTOR: RetrievalStrategy.HYBRID,
            RetrievalStrategy.KEYWORD: RetrievalStrategy.HYBRID,
            RetrievalStrategy.HYBRID: RetrievalStrategy.VECTOR,
        }
        new_strategy = strategy_fallback.get(original_strategy, RetrievalStrategy.HYBRID)
        logger.info(f"🔄 Retrying with strategy: {new_strategy.value}")

        results = self._execute_search(query, new_strategy, limit)

        if self.config.enable_rerank and len(results) > 1:
            results = rerank_results(query, results, top_k=self.config.rerank_top_k)

        return results

    def _deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        seen = {}
        for r in results:
            key = _make_dedup_key(r.payload)
            if key not in seen or r.score > seen[key].score:
                seen[key] = r
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)

    def format_results(self, results: List[RetrievalResult]) -> str:
        if not results:
            return ""

        if self.formatter:
            return self.formatter(results)

        formatted = []
        for i, r in enumerate(results, 1):
            payload = r.payload
            source_tag = f"[{r.source}]" if r.source != "vector" else ""
            formatted.append(
                f"{i}. {payload.get('name', 'Unknown')} - "
                f"{payload.get('location', '')} "
                f"{payload.get('price_tier', '')} "
                f"(score: {r.score:.3f}) {source_tag}"
            )
        return "\n".join(formatted)
