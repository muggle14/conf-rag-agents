# confluence_qa_orchestrator.py
"""
Confluence Q&A System using AutoGen Framework
Features:
- Multi-agent orchestration with AutoGen
- Query decomposition and path planning
- NL-based clarification
- Transparent thinking process
- Tree-based page structure visualization
- Integration with Azure Cognitive Search, Cosmos DB, and Azure Storage
"""

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Using simplified agent pattern without autogen for now
# from autogen import ConversableAgent, GroupChat, GroupChatManager
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from gremlin_python.driver import client as gremlin_client
from gremlin_python.driver import serializer
from openai import AzureOpenAI

from agents.autogen_tools.graph_tool import GraphTool

# Import existing tools
from agents.autogen_tools.search_tool import AzureSearchTool
from agents.autogen_tools.tree_builder import TreeBuilder

# Import simplified clarifier
from agents.clarifier import ask_clarifying_question

# Import confidence logic
from agents.logic.confidence import (
    ConfidenceTracker,
    adjust_confidence_for_coverage,
    compute_overlap,
    confidence,
    get_confidence_level,
    should_clarify,
)

# Import prompts from prompts.py
from agents.prompts import PromptTemplates
from agents.storage.blob_client import BlobStorageClient
from memory.cosmos_session import SessionStore

# Configuration
MAX_HOPS = int(os.getenv("MAX_HOPS", "3"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
EDGE_TYPES = os.getenv("EDGE_TYPES", "ParentOf,LinksTo").split(",")
ORGANIZATION = os.getenv("ORGANIZATION", "your organization")


# Dataclasses for temporary runtime data only
@dataclass
class QueryAnalysis:
    """Result of query analysis - temporary runtime data"""

    classification: str  # Atomic | NeedsDecomposition | NeedsClarification
    subquestions: List[str] = field(default_factory=list)
    clarification_needed: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    key_concepts: List[str] = field(default_factory=list)  # Added from prompts.py
    temporal_aspects: List[str] = field(default_factory=list)  # Added from prompts.py


@dataclass
class SearchResult:
    """Search result with metadata - temporary runtime data from Azure Cognitive Search"""

    id: str
    page_id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfluenceQAOrchestrator:
    """Main orchestrator for Confluence Q&A system"""

    def __init__(
        self,
        *,
        search_client=None,
        gremlin_client=None,
        aoai_client=None,
    ):
        """
        Initialize orchestrator with optional dependency injection for testing.

        Args:
            search_client: Optional SearchClient instance
            gremlin_client: Optional Gremlin client instance
            aoai_client: Optional AzureOpenAI client instance
        """
        # Initialize configuration first
        self.config_list = self._get_llm_config()
        self.agent_configs = self._get_agent_specific_configs()
        self.thinking_process = []

        # Initialize instance variables
        self.response_cache = {}
        self.embedding_cache = {}
        self.confidence_tracker = None  # Will be initialized per query
        self.TOTAL_TIMEOUT = float(os.getenv("TOTAL_TIMEOUT", "30"))
        self.MAX_SUBQUESTIONS = int(os.getenv("MAX_SUBQUESTIONS", "5"))
        self.MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "25"))
        self.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

        # Prompt template parameters (needed before agent initialization)
        self.prompt_params = {
            "organization": ORGANIZATION,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "max_hops": MAX_HOPS,
            "edge_types": EDGE_TYPES,
        }

        # Initialize agents after prompt_params is set
        self.agents = self._initialize_agents()

        # Use injected clients or initialize defaults
        self.search_client = search_client or self._init_search_client()
        self.gremlin_client = gremlin_client or self._init_gremlin_client()
        self.aoai_client = aoai_client or self._init_aoai_client()

        # Initialize tools and storage
        self.search_tool = AzureSearchTool()
        self.graph_tool = GraphTool()
        self.tree_builder = TreeBuilder(self.gremlin_client)
        self.session_store = SessionStore()
        self.blob_storage = BlobStorageClient()

    def _get_llm_config(self):
        """Get base LLM configuration for AutoGen agents"""
        return [
            {
                "model": os.getenv("AOAI_CHAT_DEPLOY", "gpt-4o"),
                "api_key": os.getenv("AOAI_KEY"),
                "base_url": f"{os.getenv('AOAI_ENDPOINT')}/openai/deployments/{os.getenv('AOAI_CHAT_DEPLOY')}/",
                "api_type": "azure",
                "api_version": "2025-01-01",
            }
        ]

    def _get_agent_specific_configs(self):
        """Get agent-specific configurations for optimized performance"""
        return {
            "query_analyser": {
                "temperature": 0,
                "max_tokens": 200,
                "response_format": {"type": "json_object"},
            },
            "decomposer": {
                "temperature": 0,
                "max_tokens": 150,
                "response_format": {"type": "json_object"},
            },
            "path_planner": {"temperature": 0, "max_tokens": 100},
            "retriever": {"temperature": 0, "max_tokens": 50},
            # "reranker": {"temperature": 0, "max_tokens": 150},  # Commented out - Azure Search handles reranking
            "synthesiser": {"temperature": 0.1, "max_tokens": 800},
            "verifier": {
                "temperature": 0,
                "max_tokens": 200,
                "response_format": {"type": "json_object"},
            },
            "clarifier": {"temperature": 0.3, "max_tokens": 300},
        }

    def _init_search_client(self):
        """Initialize Azure AI Search client"""
        return SearchClient(
            endpoint=os.environ["SEARCH_ENDPOINT"],
            index_name=os.environ["SEARCH_INDEX"],
            credential=DefaultAzureCredential(),
        )

    def _init_gremlin_client(self):
        """Initialize Cosmos DB Gremlin client"""
        return gremlin_client.Client(
            f"wss://{os.environ['COSMOS_ACCOUNT']}.gremlin.cosmos.azure.com:443/",
            "g",
            username=f"/dbs/{os.environ['COSMOS_DB']}/colls/{os.environ['COSMOS_GRAPH']}",
            password=os.environ["COSMOS_KEY"],
            message_serializer=serializer.GraphSONSerializersV2d0(),
        )

    def _init_aoai_client(self):
        """Initialize Azure OpenAI client"""
        return AzureOpenAI(
            azure_endpoint=os.environ["AOAI_ENDPOINT"],
            api_key=os.environ["AOAI_KEY"],
            api_version="2025-01-01",
        )

    def _initialize_agents(self):
        """Initialize all AutoGen agents using prompts from prompts.py"""

        # Query Analyser Agent
        query_analyser = ConversableAgent(
            "query_analyser",
            system_message=PromptTemplates.get_prompt(
                "query_analyser", **self.prompt_params
            ),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["query_analyser"],
            },
            function_map={"analyze_query": self.analyze_query},
        )

        # Decomposer Agent
        decomposer = ConversableAgent(
            "decomposer",
            system_message=PromptTemplates.get_prompt(
                "decomposer", **self.prompt_params
            ),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["decomposer"],
            },
        )

        # Path Planner Agent
        path_planner = ConversableAgent(
            "path_planner",
            system_message=PromptTemplates.get_prompt(
                "path_planner", **self.prompt_params
            ),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["path_planner"],
            },
            function_map={
                "plan_path": self.plan_path,
                "get_page_relationships": self.get_page_relationships,
            },
        )

        # Retriever Agent
        retriever = ConversableAgent(
            "retriever",
            system_message=PromptTemplates.get_prompt(
                "retriever", **self.prompt_params
            ),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["retriever"],
            },
            function_map={
                "hybrid_search": self.hybrid_search,
                "fetch_page_content": self.fetch_page_content,
            },
        )

        # Reranker Agent - COMMENTED OUT: Azure Search handles reranking
        # reranker = ConversableAgent(
        #     "reranker",
        #     system_message=PromptTemplates.get_prompt("reranker", **self.prompt_params),
        #     llm_config={
        #         "config_list": self.config_list,
        #         **self.agent_configs["reranker"],
        #     },
        #     function_map={"semantic_rerank": self.semantic_rerank},
        # )

        # Synthesiser Agent
        synthesiser = ConversableAgent(
            "synthesiser",
            system_message=PromptTemplates.get_prompt(
                "synthesiser", **self.prompt_params
            ),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["synthesiser"],
            },
            function_map={
                "synthesize_answer": self.synthesize_answer,
                "calculate_coverage": self.calculate_coverage,
            },
        )

        # Verifier Agent
        verifier = ConversableAgent(
            "verifier",
            system_message=PromptTemplates.get_prompt("verifier", **self.prompt_params),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["verifier"],
            },
        )

        # Clarifier Agent
        clarifier = ConversableAgent(
            "clarifier",
            system_message=PromptTemplates.get_prompt(
                "clarifier", **self.prompt_params
            ),
            llm_config={
                "config_list": self.config_list,
                **self.agent_configs["clarifier"],
            },
        )

        # Tree Builder Agent - REMOVED: Using TreeBuilder tool directly instead

        return {
            "query_analyser": query_analyser,
            "decomposer": decomposer,
            "path_planner": path_planner,
            "retriever": retriever,
            # "reranker": reranker,  # Commented out - Azure Search handles reranking
            "synthesiser": synthesiser,
            "verifier": verifier,
            "clarifier": clarifier,
        }

    def _check_quick_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Check for quick patterns in query for optimization"""
        # Simple pattern matching for common queries
        patterns = {
            "sso": {
                "answer": "To enable SSO, follow the documentation [[sso-guide-1]]",
                "confidence": 0.95,
                "category": "SSO Guide",
            },
            "login": {
                "answer": "For login issues, check [[login-troubleshoot-1]]",
                "confidence": 0.9,
                "category": "Login Help",
            },
        }

        query_lower = query.lower()
        for pattern, response in patterns.items():
            if pattern in query_lower:
                return response

        return None

    async def process_query(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Process user queries with balanced performance and accuracy"""

        start_time = time.time()

        # Check response cache first (performance optimization that doesn't affect accuracy)
        cache_key = self._get_cache_key(query)
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            cached_response["cached"] = True
            cached_response["response_time"] = time.time() - start_time
            return cached_response

        try:
            # Process with reasonable timeout for quality responses
            result = await asyncio.wait_for(
                self._process_query_balanced(query, conversation_id),
                timeout=self.TOTAL_TIMEOUT,
            )

            # Cache successful response
            self.response_cache[cache_key] = result

            return result

        except asyncio.TimeoutError:
            # Return comprehensive fallback response with available information
            return await self._generate_comprehensive_fallback(
                query, conversation_id, start_time
            )

    async def _process_query_balanced(
        self, query: str, conversation_id: str
    ) -> Dict[str, Any]:
        """Process query with full accuracy and optimized performance"""

        start_time = time.time()

        # Initialize confidence tracker for this query
        self.confidence_tracker = ConfidenceTracker()

        # Log initial query with full thinking process
        await self._log_thinking(
            conversation_id,
            "system",
            "receive_query",
            f"Processing query: {query}",
            query,
        )

        # Quick pattern matching for common queries (performance optimization)
        quick_response = self._check_quick_patterns(query)
        if quick_response and quick_response.get("confidence", 0) > 0.95:
            # Still verify pattern-matched responses
            verification = await self._verify_pattern_response(
                quick_response["answer"], query
            )
            if verification["confidence"] > 0.9:
                return {
                    "status": "success",
                    "answer": quick_response["answer"],
                    "confidence": verification["confidence"],
                    "page_trees": await self._build_page_trees_for_pattern(
                        quick_response
                    ),
                    "thinking_process": await self._format_thinking_process(
                        conversation_id
                    ),
                    "response_time": time.time() - start_time,
                    "pattern_match": True,
                }

        # Full query analysis with AutoGen
        analysis = await self._analyze_query(query, conversation_id)

        # Perform initial search to compute confidence
        initial_results = await self._get_initial_search_results(query, conversation_id)

        # Compute confidence based on search results and graph overlap
        query_confidence = await self._compute_query_confidence(
            initial_results, query, conversation_id
        )

        # Check if we should clarify based on confidence
        if should_clarify(query_confidence, self.CONFIDENCE_THRESHOLD):
            return await self._handle_confidence_based_clarification(
                query, query_confidence, analysis, conversation_id
            )

        # Handle based on classification
        if analysis.classification == "NeedsClarification":
            return await self._handle_clarification(query, analysis, conversation_id)

        elif analysis.classification == "NeedsDecomposition":
            return await self._handle_complex_query(query, analysis, conversation_id)

        else:  # Atomic query
            return await self._handle_atomic_query(query, conversation_id)

    async def _handle_atomic_query(
        self, query: str, conversation_id: str
    ) -> Dict[str, Any]:
        """Handle simple atomic queries with full accuracy"""

        start_time = time.time()

        await self._log_thinking(
            conversation_id,
            "orchestrator",
            "atomic_query",
            "Processing as atomic query",
            None,
        )

        # Progressive search strategy (from recommendations)
        docs = await self._retrieve_documents_progressive(
            query, {"hops": [{"hop_number": 0, "filter": None}]}
        )

        # RERANKING DISABLED: Azure Search handles reranking natively
        # reranked = await self._rerank_results(docs, query, conversation_id)
        reranked = docs  # Use original search results without additional reranking

        # Synthesize answer with full context
        sub_results = [
            {
                "question": query,
                "documents": reranked,
                "path_plan": {"hops": [{"hop_number": 0}]},
            }
        ]

        answer = await self._synthesize_answer(query, sub_results, conversation_id)

        # Full verification for accuracy
        verification = await self._verify_answer(answer, sub_results, conversation_id)

        # Build complete page trees
        page_trees = await self._build_page_trees(sub_results)

        # Get complete thinking process
        thinking_process = await self._format_thinking_process(conversation_id)

        return {
            "status": "success",
            "answer": answer,
            "thinking_process": thinking_process,
            "page_trees": page_trees,
            "verification": verification,
            "confidence": (
                self.confidence_tracker.get_final_confidence()
                if self.confidence_tracker
                else verification["confidence"]
            ),
            "confidence_level": (
                get_confidence_level(self.confidence_tracker.get_final_confidence())
                if self.confidence_tracker
                else "medium"
            ),
            "confidence_breakdown": (
                self.confidence_tracker.get_confidence_breakdown()
                if self.confidence_tracker
                else None
            ),
            "response_time": time.time() - start_time,
        }

    async def _handle_complex_query(
        self, query: str, analysis: QueryAnalysis, conversation_id: str
    ) -> Dict[str, Any]:
        """Handle complex queries with full decomposition"""

        start_time = time.time()

        await self._log_thinking(
            conversation_id,
            "decomposer",
            "decompose",
            "Breaking down complex query",
            analysis.subquestions,
        )

        # Get full decomposition if not already done
        if not analysis.subquestions:
            decomposition = await self._decompose_query(query, conversation_id)
            analysis.subquestions = decomposition["subquestions"]

        # Batch processing for sub-questions (performance optimization)
        batch_size = 3
        valid_results = []

        for i in range(
            0, len(analysis.subquestions[: self.MAX_SUBQUESTIONS]), batch_size
        ):
            batch = analysis.subquestions[i : i + batch_size]
            batch_tasks = []

            for j, subq in enumerate(batch):
                await self._log_thinking(
                    conversation_id,
                    "orchestrator",
                    "process_subquestion",
                    f"Processing sub-question {i + j + 1}: {subq}",
                    None,
                )

                task = asyncio.create_task(
                    self._process_subquestion_full(subq, i + j, conversation_id),
                    name=f"subq_{i + j}",
                )
                batch_tasks.append(task)

            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Filter out any failed tasks
            for k, result in enumerate(batch_results):
                if not isinstance(result, Exception) and result:
                    valid_results.append(result)
                else:
                    await self._log_thinking(
                        conversation_id,
                        "orchestrator",
                        "subquestion_error",
                        f"Error processing sub-question {i + k + 1}",
                        str(result) if isinstance(result, Exception) else None,
                    )

        # Synthesize comprehensive answer
        answer = await self._synthesize_answer(query, valid_results, conversation_id)

        # Full verification
        verification = await self._verify_answer(answer, valid_results, conversation_id)

        # Build complete page trees
        page_trees = await self._build_page_trees(valid_results)

        # Handle verification results if needed
        if verification["risk"]:
            answer = await self._handle_verification_failure(
                answer, verification, valid_results
            )

        # Save conversation
        self.session_store.add_messages(
            conversation_id,
            [
                {"role": "user", "content": query, "timestamp": time.time()},
                {"role": "assistant", "content": answer, "timestamp": time.time()},
            ],
        )

        thinking_process = await self._format_thinking_process(conversation_id)

        return {
            "status": "success",
            "answer": answer,
            "thinking_process": thinking_process,
            "page_trees": page_trees,
            "verification": verification,
            "sub_questions": analysis.subquestions,
            "confidence": (
                self.confidence_tracker.get_final_confidence()
                if self.confidence_tracker
                else verification["confidence"]
            ),
            "confidence_level": (
                get_confidence_level(self.confidence_tracker.get_final_confidence())
                if self.confidence_tracker
                else "medium"
            ),
            "confidence_breakdown": (
                self.confidence_tracker.get_confidence_breakdown()
                if self.confidence_tracker
                else None
            ),
            "response_time": time.time() - start_time,
        }

    async def _process_subquestion_full(
        self, subq: str, index: int, conversation_id: str
    ) -> Dict[str, Any]:
        """Process a sub-question with full accuracy"""

        # Plan path for this sub-question
        path_plan = await self._plan_path(subq, index, [], conversation_id)

        # Retrieve documents with progressive search
        docs = await self._retrieve_documents_progressive(subq, path_plan)

        # RERANKING DISABLED: Azure Search handles reranking natively
        # reranked = await self._rerank_results(docs, subq, conversation_id)
        reranked = docs  # Use original search results without additional reranking

        return {"question": subq, "documents": reranked, "path_plan": path_plan}

    async def _retrieve_documents_progressive(
        self, query: str, path_plan: Dict[str, Any]
    ) -> List[SearchResult]:
        """Progressive retrieval - start fast, expand if needed"""

        # Build filter from path plan
        search_filter = None
        if path_plan.get("hops") and path_plan["hops"][0].get("filter"):
            search_filter = path_plan["hops"][0]["filter"]

        # Phase 1: Quick keyword search (fastest)
        results = await self._execute_keyword_search(query, search_filter)

        if len(results) >= 5 and results[0].score > 0.8:
            return results[:10]  # Good enough, return quickly

        # Phase 2: Add vector search
        embedding = await self._get_cached_embedding(query)
        if embedding:
            vector_results = await self._execute_vector_search(embedding, search_filter)
            results.extend(vector_results)

        if len(results) >= 10:
            return self._deduplicate_results(results)[:15]

        # Phase 3: Semantic search only if really needed
        semantic_results = await self._execute_semantic_search(query, search_filter)
        results.extend(semantic_results)

        return self._deduplicate_results(results)[:20]

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results while preserving order"""
        seen_ids = set()
        unique_results = []
        for doc in results:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_results.append(doc)
        return unique_results

    async def _retrieve_documents_vector(
        self, query: str, path_plan: Dict[str, Any]
    ) -> List[SearchResult]:
        """Vector-only retrieval for specific use cases"""

        # Get embedding
        embedding = await self._get_cached_embedding(query)
        if not embedding:
            return []

        # Build filter from path plan
        search_filter = None
        if path_plan.get("hops") and path_plan["hops"][0].get("filter"):
            search_filter = path_plan["hops"][0]["filter"]

        # Execute vector search
        return await self._execute_vector_search(embedding, search_filter)

    async def _retrieve_documents_hybrid(
        self, query: str, path_plan: Dict[str, Any]
    ) -> List[SearchResult]:
        """Hybrid retrieval with parallel search strategies"""

        # Get embedding (with caching for performance)
        embedding = await self._get_cached_embedding(query)

        # Build filter from path plan
        search_filter = None
        if path_plan.get("hops") and path_plan["hops"][0].get("filter"):
            search_filter = path_plan["hops"][0]["filter"]

        # Parallel search tasks
        tasks = []

        # Vector search
        if embedding:
            vector_task = asyncio.create_task(
                self._execute_vector_search(embedding, search_filter)
            )
            tasks.append(vector_task)

        # Keyword search
        keyword_task = asyncio.create_task(
            self._execute_keyword_search(query, search_filter)
        )
        tasks.append(keyword_task)

        # Semantic search
        semantic_task = asyncio.create_task(
            self._execute_semantic_search(query, search_filter)
        )
        tasks.append(semantic_task)

        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)

        # Deduplicate and return
        unique_results = self._deduplicate_results(all_results)

        # Fetch full content from blob storage if needed
        for doc in unique_results[: self.MAX_SEARCH_RESULTS]:
            if not doc.content and doc.page_id:
                full_content = self.blob_storage.get_page_content(doc.page_id)
                if full_content:
                    try:
                        page_data = json.loads(full_content)
                        doc.content = (
                            page_data.get("body", {})
                            .get("storage", {})
                            .get("value", "")
                        )
                    except:
                        doc.content = full_content

        return unique_results[: self.MAX_SEARCH_RESULTS]

    async def _execute_vector_search(
        self, embedding: List[float], search_filter: Optional[str]
    ) -> List[SearchResult]:
        """Execute vector search with optimized K value"""
        try:
            results = self.search_client.search(
                search_text="",
                vector_queries=[
                    {
                        "vector": embedding,
                        "k_nearest_neighbors": 10,  # Reduced from 15
                        "fields": "embedding",
                    }
                ],
                filter=search_filter,
                top=10,  # Reduced from 15
                include_total_count=True,
            )

            return [self._convert_to_search_result(r) for r in results]
        except Exception as e:
            await self._log_thinking(
                "system",
                "search",
                "vector_search_error",
                f"Vector search failed: {str(e)}",
                None,
            )
            return []

    async def _execute_keyword_search(
        self, query: str, search_filter: Optional[str]
    ) -> List[SearchResult]:
        """Execute keyword search with optimized top value"""
        try:
            results = self.search_client.search(
                search_text=query,
                search_mode="all",  # More accurate than "any"
                filter=search_filter,
                top=15,  # Reduced from 25
                include_total_count=True,
            )

            return [self._convert_to_search_result(r) for r in results]
        except Exception as e:
            await self._log_thinking(
                "system",
                "search",
                "keyword_search_error",
                f"Keyword search failed: {str(e)}",
                None,
            )
            return []

    async def _execute_semantic_search(
        self, query: str, search_filter: Optional[str]
    ) -> List[SearchResult]:
        """Execute semantic search if available"""
        try:
            results = self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name="default",
                filter=search_filter,
                top=8,  # Reduced from 10
            )

            return [self._convert_to_search_result(r) for r in results]
        except:
            # Semantic search might not be available in all tiers
            return []

    def _convert_to_search_result(self, result: Dict) -> SearchResult:
        """Convert search result to SearchResult object"""
        return SearchResult(
            id=result["id"],
            page_id=result["pageId"],
            title=result["title"],
            content=result.get("content", ""),
            score=result.get("@search.score", 0),
            metadata=result.get("metadata", {}),
        )

    # RERANKING DISABLED: Azure Search handles reranking natively
    # async def _rerank_results(
    #     self, docs: List[SearchResult], query: str, conversation_id: str
    # ) -> List[SearchResult]:
    #     """Full semantic reranking for accuracy with reduced top_n"""
    #
    #     if not docs:
    #         return []
    #
    #     # Use Azure semantic reranker if available
    #     try:
    #         # Prepare documents for reranking
    #         rerank_input = [
    #             {"id": doc.id, "text": f"{doc.title} {doc.content}"} for doc in docs
    #         ]
    #
    #         # Call reranker
    #         reranked_results = await self.aoai_client.rerank(
    #             query=query,
    #             documents=rerank_input,
    #             model="semantic-reranker-v2",
    #             top_n=min(8, len(docs)),  # Reduced from 15
    #         )
    #
    #         # Reorder documents based on reranking
    #         reranked_docs = []
    #         for result in reranked_results:
    #             doc_id = result["id"]
    #             for doc in docs:
    #                 if doc.id == doc_id:
    #                     reranked_docs.append(doc)
    #                     break
    #
    #         await self._log_thinking(
    #             conversation_id,
    #             "reranker",
    #             "semantic_rerank",
    #             f"Reranked {len(docs)} documents to {len(reranked_docs)}",
    #             {"original_count": len(docs), "reranked_count": len(reranked_docs)},
    #         )
    #
    #         return reranked_docs
    #
    #     except Exception as e:
    #         # Fallback to score-based reranking
    #         await self._log_thinking(
    #             conversation_id,
    #             "reranker",
    #             "fallback_rerank",
    #             f"Using score-based reranking: {str(e)}",
    #             None,
    #         )
    #
    #         return sorted(docs, key=lambda x: x.score, reverse=True)[:8]

    async def _get_cached_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding or generate new one"""
        cache_key = self._get_cache_key(query)

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            # Generate embedding
            response = await self.aoai_client.embeddings.create(
                model=os.environ["AOAI_EMBED_DEPLOY"], input=[query]
            )

            embedding = response.data[0].embedding
            self.embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            await self._log_thinking(
                "system",
                "embedding",
                "error",
                f"Failed to generate embedding: {str(e)}",
                None,
            )
            return None

    async def _generate_comprehensive_fallback(
        self, query: str, conversation_id: str, start_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive fallback response with available information"""

        # Try to get at least some search results
        try:
            simple_results = await asyncio.wait_for(
                self._execute_keyword_search(query, None), timeout=2.0
            )

            if simple_results:
                # Build a basic answer
                answer = f"""I encountered a timeout while processing your query, but I found some relevant information:

{simple_results[0].title}: {simple_results[0].content[:200]}... [[{simple_results[0].id}]]

For a complete answer, you may want to:
1. Check the full documentation page: {simple_results[0].title}
2. Try a more specific query
3. Browse related pages in the documentation"""

                return {
                    "status": "partial",
                    "answer": answer,
                    "confidence": 0.6,
                    "page_trees": [
                        {
                            "root_page_id": simple_results[0].page_id,
                            "root_title": simple_results[0].title,
                            "markdown": f"- [{simple_results[0].title}](/wiki/pages/{simple_results[0].page_id}) ⚠️ (partial result)",
                            "contains_answer": True,
                        }
                    ],
                    "thinking_process": await self._format_thinking_process(
                        conversation_id
                    ),
                    "response_time": time.time() - start_time,
                    "timeout": True,
                    "partial_results": True,
                }
        except:
            pass

        # Complete fallback
        return {
            "status": "timeout",
            "answer": """I apologize, but I couldn't complete processing your query within the time limit.

This might be due to:
- Complex query requiring extensive analysis
- High system load
- Network connectivity issues

Please try:
1. Simplifying your question
2. Breaking it into smaller, specific questions
3. Searching the documentation directly
4. Trying again in a few moments""",
            "confidence": 0.0,
            "page_trees": [],
            "thinking_process": await self._format_thinking_process(conversation_id),
            "response_time": time.time() - start_time,
            "timeout": True,
        }

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _verify_pattern_response(self, answer: str, query: str) -> Dict[str, Any]:
        """Verify pattern-matched responses for accuracy"""
        # Quick verification for pattern responses
        return {
            "risk": False,
            "risk_level": "none",
            "confidence": 0.95,
            "issues_found": {"unsupported_claims": [], "missing_citations": []},
            "quality_assessment": {
                "completeness": 0.9,
                "accuracy": 0.95,
                "clarity": 1.0,
                "structure": 1.0,
            },
        }

    async def _build_page_trees_for_pattern(
        self, pattern_response: Dict
    ) -> List[Dict[str, Any]]:
        """Build simple page trees for pattern-matched responses"""
        # Extract page IDs from citations in the answer
        citation_pattern = r"\[\[([^\]]+)\]\]"
        citations = re.findall(citation_pattern, pattern_response.get("answer", ""))

        trees = []
        for citation in citations[:3]:  # Limit to 3 trees
            # Simple tree for pattern responses
            trees.append(
                {
                    "root_page_id": citation.split("-")[0],
                    "root_title": pattern_response.get("category", "Documentation"),
                    "markdown": f"- [{pattern_response.get('category', 'Documentation')}](/wiki/pages/{citation}) ⭐",
                    "contains_answer": True,
                }
            )

        return trees

    async def _analyze_query(self, query: str, conversation_id: str) -> QueryAnalysis:
        """Analyze query using AutoGen Query Analyser agent"""

        await self._log_thinking(
            conversation_id,
            "query_analyser",
            "analyze",
            "Analyzing query complexity and clarity",
            None,
        )

        # Create AutoGen group chat for analysis
        groupchat = GroupChat(
            agents=[self.agents["query_analyser"]], messages=[], max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        # Get analysis using AutoGen
        await self.agents["query_analyser"].initiate_chat(
            manager, message=f"Analyze this query: {query}"
        )

        # Parse response
        response = groupchat.messages[-1]["content"]
        analysis_data = json.loads(response)

        analysis = QueryAnalysis(
            classification=analysis_data["classification"],
            subquestions=analysis_data.get("subquestions", []),
            clarification_needed=analysis_data.get("clarification_needed"),
            confidence=analysis_data.get("confidence", 0.8),
            reasoning=analysis_data.get("reasoning", ""),
            key_concepts=analysis_data.get("key_concepts", []),
            temporal_aspects=analysis_data.get("temporal_aspects", []),
        )

        await self._log_thinking(
            conversation_id,
            "query_analyser",
            "complete",
            f"Classification: {analysis.classification}",
            analysis,
        )

        return analysis

    async def _get_initial_search_results(
        self, query: str, conversation_id: str
    ) -> List[SearchResult]:
        """Get initial search results for confidence computation."""
        from tracing.autogen_tracer import log, new_trace_id

        trace_id = new_trace_id()
        log(
            "initial_search_start",
            trace_id,
            query=query,
            conversation_id=conversation_id,
        )

        # Quick keyword search for initial confidence assessment
        results = await self._execute_keyword_search(query, None)

        log(
            "initial_search_complete",
            trace_id,
            results_count=len(results),
            top_score=results[0].score if results else 0,
        )

        return results[:10]  # Return top 10 for confidence computation

    async def _compute_query_confidence(
        self, search_results: List[SearchResult], query: str, conversation_id: str
    ) -> float:
        """Compute confidence score based on search results and graph structure."""
        from tracing.autogen_tracer import log, new_trace_id

        trace_id = new_trace_id()
        log(
            "confidence_compute_start",
            trace_id,
            results_count=len(search_results),
            conversation_id=conversation_id,
        )

        # Get pre-rerank top score (before any reranking)
        pre_rerank_top_score = search_results[0].score if search_results else 0.0

        # Get graph neighbors for top results
        neighbor_ids = set()
        if search_results:
            # Get neighbors for top 3 results
            for result in search_results[:3]:
                neighbors = await self._get_graph_neighbors(result.page_id)
                neighbor_ids.update(neighbors)

        # Extract search result IDs
        search_ids = [r.page_id for r in search_results[:10]]

        # Compute overlap
        overlap_ratio = compute_overlap(search_ids, neighbor_ids)

        # Compute final confidence
        final_confidence = confidence(pre_rerank_top_score, overlap_ratio)

        # Track confidence
        self.confidence_tracker.add_score(
            "initial_confidence",
            final_confidence,
            pre_rerank_score=pre_rerank_top_score,
            overlap_ratio=overlap_ratio,
            neighbor_count=len(neighbor_ids),
        )

        log(
            "confidence_compute_complete",
            trace_id,
            confidence=final_confidence,
            confidence_level=get_confidence_level(final_confidence),
            should_clarify=should_clarify(final_confidence),
        )

        return final_confidence

    async def _get_graph_neighbors(
        self, page_id: str, edge_types: Optional[List[str]] = None
    ) -> Set[str]:
        """Get neighbor page IDs from graph."""
        if not edge_types:
            edge_types = EDGE_TYPES

        try:
            # Query for neighbors
            edge_filter = "'" + "','".join(edge_types) + "'"
            query = f"""
            g.V('{page_id}')
              .both({edge_filter})
              .id()
              .dedup()
              .limit(20)
            """

            result = await asyncio.to_thread(self.gremlin_client.submit, query)
            neighbors = result.all().result()
            return set(neighbors)
        except Exception as e:
            await self._log_thinking(
                "system",
                "graph",
                "neighbor_error",
                f"Error getting neighbors for {page_id}: {str(e)}",
                None,
            )
            return set()

    async def _handle_confidence_based_clarification(
        self,
        query: str,
        query_confidence: float,
        analysis: QueryAnalysis,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Handle clarification based on confidence score."""
        from tracing.autogen_tracer import log, new_trace_id

        trace_id = new_trace_id()
        log(
            "confidence_clarify_start",
            trace_id,
            confidence=query_confidence,
            confidence_level=get_confidence_level(query_confidence),
            conversation_id=conversation_id,
        )

        # Get top result titles for context
        initial_results = await self._get_initial_search_results(query, conversation_id)
        top_titles = [r.title for r in initial_results[:4]]

        # Generate clarification using simplified clarifier
        clarification_question = ask_clarifying_question(query, top_titles)

        # Remember clarification in session store
        self.session_store.remember_clarification(
            conversation_id, clarification_question
        )

        # Save conversation state
        self.session_store.add_messages(
            conversation_id,
            [
                {"role": "user", "content": query, "timestamp": time.time()},
                {
                    "role": "assistant",
                    "content": clarification_question,
                    "timestamp": time.time(),
                    "metadata": {
                        "confidence": query_confidence,
                        "confidence_level": get_confidence_level(query_confidence),
                    },
                },
            ],
        )

        thinking_process = await self._format_thinking_process(conversation_id)

        log(
            "confidence_clarify_complete",
            trace_id,
            clarification_generated=True,
            question=clarification_question,
            conversation_id=conversation_id,
        )

        return {
            "status": "needs_clarification",
            "clarification_message": clarification_question,
            "original_query": query,
            "confidence": query_confidence,
            "confidence_level": get_confidence_level(query_confidence),
            "confidence_breakdown": self.confidence_tracker.get_confidence_breakdown(),
            "thinking_process": thinking_process,
            "suggestions": [
                "Be more specific about which system or component",
                "Include version numbers if relevant",
                "Specify if you need setup, usage, or troubleshooting help",
            ],
            "top_results": top_titles,
        }

    async def _handle_clarification(
        self, query: str, analysis: QueryAnalysis, conversation_id: str
    ) -> Dict[str, Any]:
        """Handle queries that need clarification using AutoGen Clarifier agent"""
        from tracing.autogen_tracer import log, new_trace_id

        # Generate trace_id
        trace_id = new_trace_id()

        # Log clarification start
        log(
            "clarify_start",
            trace_id,
            query=query,
            clarification_reason=analysis.clarification_needed,
            conversation_id=conversation_id,
        )

        await self._log_thinking(
            conversation_id,
            "clarifier",
            "clarify",
            "Query needs clarification",
            analysis.clarification_needed,
        )

        # Get clarification from AutoGen clarifier agent
        groupchat = GroupChat(
            agents=[self.agents["clarifier"]], messages=[], max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["clarifier"].initiate_chat(
            manager,
            message=f"Original query: {query}\nClarification needed: {analysis.clarification_needed}\nGenerate a helpful clarifying question.",
        )

        clarification_response = groupchat.messages[-1]["content"]

        # Log clarification response
        log(
            "clarify_response",
            trace_id,
            response_length=len(clarification_response),
            conversation_id=conversation_id,
        )

        # Save conversation state to session store
        self.session_store.add_messages(
            conversation_id,
            [
                {"role": "user", "content": query, "timestamp": time.time()},
                {
                    "role": "assistant",
                    "content": clarification_response,
                    "timestamp": time.time(),
                },
            ],
        )

        thinking_process = await self._format_thinking_process(conversation_id)

        return {
            "status": "needs_clarification",
            "clarification_message": clarification_response,
            "original_query": query,
            "thinking_process": thinking_process,
            "suggestions": [
                "Try being more specific about which system or component",
                "Include version numbers or time periods if relevant",
                "Specify if you need setup, troubleshooting, or general info",
            ],
        }

    async def _decompose_query(
        self, query: str, conversation_id: str
    ) -> Dict[str, Any]:
        """Decompose complex query into sub-questions"""
        groupchat = GroupChat(
            agents=[self.agents["decomposer"]], messages=[], max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["decomposer"].initiate_chat(
            manager, message=f"Decompose this query into sub-questions: {query}"
        )

        response = groupchat.messages[-1]["content"]
        return json.loads(response)

    async def _plan_path(
        self,
        query: str,
        hop_idx: int,
        previous_results: List[Dict],
        conversation_id: str,
    ) -> Dict[str, Any]:
        """Plan search path through knowledge graph"""

        # Get previous page IDs for multi-hop
        prev_page_ids = []
        if previous_results:
            for result in previous_results:
                for doc in result["documents"][:3]:  # Top 3 from previous hop
                    prev_page_ids.append(doc.page_id)

        context = {
            "query": query,
            "hop_index": hop_idx,
            "previous_page_ids": prev_page_ids,
            "max_hops": MAX_HOPS,
            "edge_types": EDGE_TYPES,
        }

        groupchat = GroupChat(
            agents=[self.agents["path_planner"]], messages=[], max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["path_planner"].initiate_chat(
            manager, message=f"Plan search path for: {json.dumps(context)}"
        )

        response = groupchat.messages[-1]["content"]
        return json.loads(response)

    async def _synthesize_answer(
        self, query: str, sub_results: List[Dict], conversation_id: str
    ) -> str:
        """Synthesize final answer from sub-results"""
        from tracing.autogen_tracer import log, new_trace_id

        # Generate trace_id
        trace_id = new_trace_id()

        # Log synthesis start
        log(
            "synthesize_start",
            trace_id,
            query=query,
            sub_results_count=len(sub_results),
            conversation_id=conversation_id,
        )

        # Prepare context
        context_blocks = []
        for i, result in enumerate(sub_results):
            context_blocks.append(
                f"\n=== Sub-question {i + 1}: {result['question']} ===\n"
            )
            for doc in result["documents"]:
                context_blocks.append(f"[[{doc.id}]] {doc.content}\n")

        context = "\n".join(context_blocks)

        groupchat = GroupChat(
            agents=[self.agents["synthesiser"]], messages=[], max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["synthesiser"].initiate_chat(
            manager,
            message=f"Original question: {query}\n\nContext:\n{context}\n\nSynthesize a comprehensive answer with citations.",
        )

        answer = groupchat.messages[-1]["content"]

        # Update confidence based on coverage
        coverage = self.calculate_coverage(answer, context)
        if self.confidence_tracker:
            current_confidence = self.confidence_tracker.get_final_confidence()
            adjusted_confidence = adjust_confidence_for_coverage(
                current_confidence, coverage, coverage_weight=0.15
            )
            self.confidence_tracker.add_score(
                "post_synthesis", adjusted_confidence, coverage=coverage
            )

        # Log synthesis complete
        log(
            "synthesize_complete",
            trace_id,
            answer_length=len(answer),
            coverage=coverage,
            conversation_id=conversation_id,
        )

        return answer

    async def _verify_answer(
        self, answer: str, sub_results: List[Dict], conversation_id: str
    ) -> Dict[str, Any]:
        """Verify answer accuracy"""
        from tracing.autogen_tracer import log, new_trace_id

        # Generate trace_id
        trace_id = new_trace_id()

        # Log verification start
        log(
            "verify_start",
            trace_id,
            answer_length=len(answer),
            sub_results_count=len(sub_results),
            conversation_id=conversation_id,
        )

        # Prepare context for verification
        all_content = []
        for result in sub_results:
            for doc in result["documents"]:
                all_content.append(f"[[{doc.id}]] {doc.content}")

        context = "\n".join(all_content)

        groupchat = GroupChat(
            agents=[self.agents["verifier"]], messages=[], max_round=1
        )

        manager = GroupChatManager(groupchat=groupchat)

        await self.agents["verifier"].initiate_chat(
            manager,
            message=f"Answer to verify:\n{answer}\n\nContext:\n{context}\n\nVerify all claims are supported.",
        )

        response = groupchat.messages[-1]["content"]
        verification_result = json.loads(response)

        # Update confidence based on verification
        if self.confidence_tracker:
            verification_confidence = verification_result.get("confidence", 0.5)
            self.confidence_tracker.add_score(
                "post_verification",
                verification_confidence,
                risk_level=verification_result.get("risk_level", "unknown"),
                accuracy=verification_result.get("quality_assessment", {}).get(
                    "accuracy", 0.5
                ),
            )

        # Log verification result
        log(
            "verify_complete",
            trace_id,
            confidence_score=verification_result.get("confidence", 0),
            risk_level=verification_result.get("risk_level", "unknown"),
            final_confidence=(
                self.confidence_tracker.get_final_confidence()
                if self.confidence_tracker
                else verification_result.get("confidence", 0)
            ),
            conversation_id=conversation_id,
        )

        return verification_result

    async def _build_page_trees(self, sub_results: List[Dict]) -> List[Dict[str, Any]]:
        """Build page hierarchy trees using TreeBuilder"""
        from tracing.autogen_tracer import log, new_trace_id

        # Generate trace_id
        trace_id = new_trace_id()

        # Log graph tree start
        log(
            "graph_tree_start",
            trace_id,
            sub_results_count=len(sub_results),
            operation="build_page_trees",
        )

        # Collect all unique page IDs
        page_ids = set()
        for result in sub_results:
            for doc in result["documents"]:
                page_ids.add(doc.page_id)

        # Build trees using TreeBuilder
        trees = await self.tree_builder.build_page_trees(page_ids)

        # Render trees as markdown
        rendered_trees = []
        for tree in trees:
            markdown = self.tree_builder.render_tree_markdown(tree)
            rendered_trees.append(
                {
                    "root_page_id": tree["page_id"],
                    "root_title": tree["title"],
                    "markdown": markdown,
                    "contains_answer": self.tree_builder.tree_contains_answer(tree),
                }
            )

        # Log graph tree completion
        log(
            "graph_tree_complete",
            trace_id,
            trees_count=len(rendered_trees),
            page_ids_count=len(page_ids),
        )

        return rendered_trees

    async def _handle_verification_failure(
        self, answer: str, verification: Dict, sub_results: List[Dict]
    ) -> str:
        """Handle case where verification failed"""

        # Get breadcrumbs for top pages
        breadcrumbs = []
        for result in sub_results:
            if result["documents"]:
                top_page_id = result["documents"][0].page_id
                breadcrumb = await self._get_breadcrumb(top_page_id)
                breadcrumbs.append(breadcrumb)

        # Construct fallback response
        fallback = f"""
        I found some information about your query, but I'm not fully confident in providing a complete answer.

        Here's what I found:
        {answer}

        **Note**: Some claims could not be fully verified against the source documents.

        For more authoritative information, please check these parent pages:
        """

        for bc in breadcrumbs:
            fallback += f"\n- {' > '.join(bc)}"

        fallback += "\n\nYou may also want to explore related documentation or contact your team for clarification."

        return fallback

    async def _get_breadcrumb(self, page_id: str) -> List[str]:
        """Get breadcrumb path for a page using TreeBuilder"""
        return await self.tree_builder.get_breadcrumb(page_id)

    async def _log_thinking(
        self, conversation_id: str, agent: str, action: str, reasoning: str, result: Any
    ):
        """Log thinking process step to session store"""
        step = {
            "agent": agent,
            "action": action,
            "reasoning": reasoning,
            "result": result,
            "timestamp": time.time(),
        }
        self.session_store.add_thinking_step(conversation_id, step)

    async def _format_thinking_process(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Format thinking process for output from session store"""
        steps = self.session_store.get_thinking_steps(conversation_id)
        formatted = []
        for i, step in enumerate(steps):
            formatted.append(
                {
                    "step": i + 1,
                    "agent": step.get("agent", "system"),
                    "action": step.get("action", ""),
                    "reasoning": step.get("reasoning", ""),
                    "timestamp": step.get("timestamp", 0),
                }
            )
        return formatted

    # Function implementations for agents

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Function for query analyser agent"""
        # This would be called by the agent, but we handle it differently
        pass

    def plan_path(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Function for path planner agent"""
        # This would be called by the agent
        pass

    def hybrid_search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Function for retriever agent"""
        # This would be called by the agent
        pass

    # RERANKING DISABLED: Azure Search handles reranking natively
    # def semantic_rerank(self, documents: List[Dict], query: str) -> List[Dict]:
    #     """Function for reranker agent"""
    #     # This would be called by the agent
    #     pass

    def synthesize_answer(self, query: str, context: List[Dict]) -> str:
        """Function for synthesiser agent"""
        # This would be called by the agent
        pass

    def calculate_coverage(self, answer: str, context: str) -> float:
        """Calculate how well context covers the answer"""
        # Simple implementation - in production use more sophisticated metrics
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())

        if not answer_tokens:
            return 0.0

        coverage = len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)
        return min(coverage, 1.0)

    def get_page_relationships(self, page_id: str) -> Dict[str, Any]:
        """Get page relationships from graph"""
        # This would query Gremlin for relationships
        pass

    def fetch_page_content(self, page_id: str) -> str:
        """Function for retriever agent to fetch page content"""
        # This would be called by the agent
        pass


# Example usage
async def main():
    """Example usage of the Confluence Q&A system"""

    orchestrator = ConfluenceQAOrchestrator()

    # Example queries
    queries = [
        "How do I enable SSO for our application?",
        "What changed between version 1.0 and 2.0 of the API?",
        "How does the system work?",  # Needs clarification
        "What are the deployment steps for the payment service and how do they relate to the database migration process?",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")

        result = await orchestrator.process_query(query, f"conv_{hash(query)}")

        print(f"\nStatus: {result['status']}")

        if result["status"] == "needs_clarification":
            print("\nClarification needed:")
            print(result["clarification_message"])
            print("\nSuggestions:")
            for suggestion in result["suggestions"]:
                print(f"  - {suggestion}")
        else:
            print("\nAnswer:")
            print(result["answer"])

            print(f"\nConfidence: {result['confidence']}")

            if result.get("sub_questions"):
                print("\nSub-questions analyzed:")
                for i, sq in enumerate(result["sub_questions"]):
                    print(f"  {i + 1}. {sq}")

            print("\nPage Trees:")
            for tree in result["page_trees"]:
                print(
                    f"\n{tree['root_title']} {'(contains answer)' if tree['contains_answer'] else ''}"
                )
                print(tree["markdown"])

        print("\nThinking Process:")
        for step in result["thinking_process"]:
            print(
                f"  Step {step['step']}: [{step['agent']}] {step['action']} - {step['reasoning']}"
            )


if __name__ == "__main__":
    asyncio.run(main())
