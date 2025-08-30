# Performance Optimization Guide for Confluence Q&A System

## Overview
This guide provides performance optimization strategies for the Confluence Q&A system that maintain accuracy and completeness while improving response times.

## Performance Philosophy: Accuracy First, Speed Second

### Key Principles
1. **Never compromise answer quality for speed**
2. **Use caching and parallel processing for performance**
3. **Maintain full verification and thinking processes**
4. **Provide complete page trees and citations**

## 1. Safe Performance Optimizations

### 1.1 Multi-Level Caching (No Accuracy Impact)
**Implementation**: Three-tier caching system
```python
class MultiLevelCache:
    """Multi-level caching for faster responses without quality loss"""

    def __init__(self):
        # L1: In-memory cache (< 1ms access)
        self.memory_cache = TTLCache(maxsize=1000, ttl=3600)

        # L2: Redis cache (< 10ms access)
        self.redis = aioredis.from_url(
            "redis://localhost",
            encoding="utf-8",
            decode_responses=True,
            max_connections=50
        )

        # L3: Cosmos DB cache (< 50ms access)
        self.cosmos_cache = CosmosCache()
```

**Benefits**:
- Instant responses for repeated queries
- No impact on answer quality
- Reduces load on Azure services

### 1.2 Parallel Processing (Improves Speed)
**Implementation**: Execute multiple operations concurrently
```python
async def parallel_search_strategies(self, query: str):
    """Execute multiple search strategies in parallel"""

    tasks = [
        self._vector_search(query),
        self._keyword_search(query),
        self._semantic_search(query)
    ]

    # Execute all searches concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge and deduplicate results
    return self._merge_search_results(results)
```

**Benefits**:
- 40-60% reduction in search latency
- More comprehensive results
- Better coverage of different query types

### 1.3 Connection Pooling (Infrastructure)
**Implementation**: Pre-establish connections to Azure services
```python
# Cosmos DB connection pooling
connection_config = {
    "ConnectionMode": "Direct",
    "ConnectionProtocol": "Tcp",
    "MaxConnectionLimit": 100,
    "IdleConnectionTimeout": 120,
    "EnableEndpointDiscovery": True
}

# Search client with keep-alive
search_client_config = {
    "connection_timeout": 10,
    "read_timeout": 30,
    "max_retries": 3,
    "retry_on_timeout": True
}
```

**Benefits**:
- Eliminates connection overhead
- More stable performance
- Better resource utilization

### 1.4 Embedding Cache (Computation Savings)
**Implementation**: Cache embeddings for common queries
```python
class EmbeddingCache:
    """Cache embeddings to avoid recomputation"""

    def __init__(self):
        self.cache = {}
        self.preload_common_queries()

    async def get_or_compute(self, text: str) -> List[float]:
        cache_key = self._get_key(text)

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Compute and cache
        embedding = await self._compute_embedding(text)
        self.cache[cache_key] = embedding
        return embedding
```

**Benefits**:
- Saves 200-300ms per cached query
- Reduces Azure OpenAI API calls
- Deterministic results

### 1.5 Smart Query Patterns (Fast Path)
**Implementation**: Pre-built responses for common queries with verification
```python
class QueryPatternMatcher:
    """Match common patterns with verification"""

    async def match_and_verify(self, query: str) -> Optional[Dict]:
        # Pattern matching
        pattern_result = self._match_pattern(query)
        if not pattern_result:
            return None

        # Always verify pattern responses
        verification = await self._verify_response(
            pattern_result['answer'],
            pattern_result['sources']
        )

        if verification['confidence'] > 0.9:
            return pattern_result

        return None  # Fall back to full search
```

**Benefits**:
- Instant responses for common queries
- Maintains accuracy through verification
- Reduces load on complex pipeline

## 2. Agent Optimization (Balanced)

### 2.1 Agent Configuration
```python
# Balanced configuration for accuracy and speed
AGENT_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0,      # Deterministic
    "max_tokens": 500,     # Full responses
    "top_p": 0.1,
    "seed": 42,           # Caching
    "stream": True,       # Perceived speed
    "timeout": 5.0        # Reasonable timeout
}

# Agent-specific timeouts
AGENT_TIMEOUTS = {
    "query_analyser": 1.0,
    "decomposer": 2.0,
    "path_planner": 1.5,
    "retriever": 3.0,
    "reranker": 2.0,
    "synthesiser": 3.0,
    "verifier": 2.0,      # Full verification
    "tree_builder": 2.0   # Complete trees
}
```

### 2.2 Agent Pool Pre-warming
```python
class AgentPool:
    """Pre-warmed agent pool for faster startup"""

    def __init__(self, size: int = 5):
        self.agents = []
        self._initialize_pool(size)

    async def _initialize_pool(self, size: int):
        """Pre-initialize agents during startup"""
        for _ in range(size):
            agent = await self._create_agent()
            self.agents.append(agent)

    async def get_agent(self) -> AutoGenAgent:
        """Get pre-warmed agent"""
        if self.agents:
            return self.agents.pop()
        return await self._create_agent()
```

## 3. Azure Service Optimization

### 3.1 Azure Cognitive Search
```python
# Optimized search configuration
SEARCH_CONFIG = {
    "search_mode": "all",           # Accurate results
    "query_type": "semantic",       # Better relevance
    "semantic_configuration": "default",
    "speller": "lexicon",          # Handle typos
    "top": 15,                     # Sufficient results
    "include_total_count": True,
    "facets": ["category", "lastModified"],
    "highlight_fields": ["content", "title"]
}

# Parallel search strategies
async def comprehensive_search(self, query: str):
    """Execute multiple search types in parallel"""

    tasks = [
        # Vector search for semantic similarity
        self._vector_search(query, k=15),

        # Keyword search for exact matches
        self._keyword_search(query, mode="all"),

        # Fuzzy search for typos
        self._fuzzy_search(query, distance=2)
    ]

    results = await asyncio.gather(*tasks)
    return self._merge_and_rank(results)
```

### 3.2 Cosmos DB Optimization
```python
# Query optimization with proper indexing
INDEXING_POLICY = {
    "automatic": True,
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "compositeIndexes": [
        [
            {"path": "/conversationId", "order": "ascending"},
            {"path": "/timestamp", "order": "descending"}
        ]
    ]
}

# Batch operations for efficiency
async def batch_save_thinking_steps(self, steps: List[Dict]):
    """Batch save thinking steps"""

    # Group by partition key
    grouped = defaultdict(list)
    for step in steps:
        grouped[step['conversationId']].append(step)

    # Batch operations per partition
    tasks = []
    for partition_key, items in grouped.items():
        task = self._batch_upsert(partition_key, items)
        tasks.append(task)

    await asyncio.gather(*tasks)
```

### 3.3 Azure OpenAI Optimization
```python
# Efficient embedding generation
class EmbeddingService:
    """Optimized embedding service"""

    def __init__(self):
        self.batch_size = 16  # Max batch size
        self.cache = EmbeddingCache()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with caching and batching"""

        # Check cache first
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Batch generate uncached embeddings
        if uncached_texts:
            new_embeddings = await self._batch_generate(uncached_texts)

            # Cache and add to results
            for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                self.cache.set(text, embedding)
                results.append((idx, embedding))

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]
```

## 4. Query Processing Pipeline

### 4.1 Optimized Pipeline Flow
```python
async def process_query_optimized(self, query: str) -> Dict:
    """Optimized query processing with full accuracy"""

    # Stage 1: Cache check (< 10ms)
    cached = await self.cache.get(query)
    if cached:
        return cached

    # Stage 2: Pattern matching with verification (< 100ms)
    pattern_result = await self.pattern_matcher.match_and_verify(query)
    if pattern_result:
        return pattern_result

    # Stage 3: Parallel analysis and search (< 1s)
    analysis_task = self.analyze_query(query)
    initial_search_task = self.quick_search(query)

    analysis, initial_results = await asyncio.gather(
        analysis_task,
        initial_search_task
    )

    # Stage 4: Full processing based on analysis
    if analysis.classification == "NeedsDecomposition":
        # Process sub-questions in parallel
        result = await self.process_complex_query(query, analysis)
    else:
        # Process atomic query with full pipeline
        result = await self.process_atomic_query(query, initial_results)

    # Stage 5: Cache and return
    await self.cache.set(query, result)
    return result
```

### 4.2 Progressive Response Strategy
```python
class ProgressiveResponse:
    """Send initial response quickly, enhance progressively"""

    async def respond_progressively(self, query: str, stream):
        # Quick initial response
        initial = await self.get_quick_answer(query)
        await stream.send({
            "status": "initial",
            "answer": initial,
            "confidence": 0.7
        })

        # Enhanced response with verification
        enhanced = await self.get_full_answer(query)
        await stream.send({
            "status": "enhanced",
            "answer": enhanced['answer'],
            "confidence": enhanced['confidence'],
            "sources": enhanced['sources']
        })

        # Final response with trees and verification
        final = await self.build_complete_response(enhanced)
        await stream.send({
            "status": "final",
            "answer": final['answer'],
            "confidence": final['confidence'],
            "page_trees": final['trees'],
            "verification": final['verification']
        })
```

## 5. Monitoring and Metrics

### 5.1 Performance Tracking
```python
PERFORMANCE_METRICS = {
    "target_response_time": 5000,  # 5 seconds target
    "target_p95": 8000,            # 8 seconds for 95th percentile
    "min_confidence": 0.7,         # Minimum confidence threshold
    "cache_hit_target": 0.4,       # 40% cache hit rate
    "parallel_efficiency": 0.8     # 80% parallel efficiency
}

# Track without compromising accuracy
class PerformanceMonitor:
    """Monitor performance while maintaining quality"""

    async def track_query(self, query: str, result: Dict):
        metrics = {
            "response_time": result['response_time'],
            "confidence": result['confidence'],
            "cache_hit": result.get('cached', False),
            "verification_passed": result['verification']['risk'] == False,
            "complete_trees": len(result.get('page_trees', [])) > 0,
            "thinking_process": len(result.get('thinking_process', [])) > 0
        }

        # Alert only on quality issues
        if metrics['confidence'] < 0.7:
            await self.alert("Low confidence response", metrics)

        if not metrics['verification_passed']:
            await self.alert("Verification failed", metrics)
```

## 6. Best Practices

### Do's ✅
1. **Use caching aggressively** - Doesn't affect accuracy
2. **Parallelize independent operations** - Faster without quality loss
3. **Pre-compute common embeddings** - Save computation time
4. **Pool connections** - Reduce overhead
5. **Stream responses** - Better perceived performance
6. **Monitor quality metrics** - Ensure accuracy maintained

### Don'ts ❌
1. **Don't skip verification** - Critical for accuracy
2. **Don't limit search results too much** - May miss relevant docs
3. **Don't reduce token limits** - Incomplete responses
4. **Don't skip thinking process** - Users need transparency
5. **Don't simplify page trees** - Full hierarchy important
6. **Don't use aggressive timeouts** - May timeout valid queries

## 7. Performance Expectations

### Realistic Targets
- **Simple queries (cached)**: 50-200ms
- **Simple queries (uncached)**: 2-4 seconds
- **Complex queries**: 4-8 seconds
- **Very complex queries**: 8-12 seconds

### Quality Metrics
- **Answer confidence**: > 0.8
- **Verification pass rate**: > 95%
- **Citation accuracy**: 100%
- **Tree completeness**: 100%

## Conclusion

The optimized system achieves good performance through:
1. **Intelligent caching** at multiple levels
2. **Parallel processing** where possible
3. **Connection pooling** and pre-warming
4. **Smart pattern matching** with verification
5. **Progressive responses** for better UX

All while maintaining:
- Full answer accuracy
- Complete verification
- Detailed thinking processes
- Comprehensive page trees
- Proper citations

This balanced approach ensures users get fast, accurate, and complete responses.
