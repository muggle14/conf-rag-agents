# monitoring.py
"""
Monitoring and observability for Confluence Q&A system
Integrates with Azure Application Insights and custom metrics
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

from applicationinsights import TelemetryClient
from opencensus.ext.azure import metrics_exporter
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module

logger = logging.getLogger(__name__)


class MetricsManager:
    """Manages custom metrics for the Q&A system"""

    def __init__(self):
        # Initialize Application Insights
        self.instrumentation_key = os.getenv("APPLICATIONINSIGHTS_INSTRUMENTATION_KEY")
        self.telemetry_client = TelemetryClient(self.instrumentation_key)

        # Initialize OpenCensus metrics
        self.stats = stats_module.stats
        self.view_manager = self.stats.view_manager

        # Set up Azure metrics exporter
        if self.instrumentation_key:
            exporter = metrics_exporter.new_metrics_exporter(
                connection_string=f"InstrumentationKey={self.instrumentation_key}"
            )
            self.view_manager.register_exporter(exporter)

        # Define measures
        self.query_latency_ms = measure_module.MeasureFloat(
            "query_latency_ms", "Query processing latency in milliseconds", "ms"
        )

        self.agent_latency_ms = measure_module.MeasureFloat(
            "agent_latency_ms", "Individual agent processing latency", "ms"
        )

        self.confidence_score = measure_module.MeasureFloat(
            "confidence_score", "Answer confidence score", "score"
        )

        self.token_count = measure_module.MeasureInt(
            "token_count", "Token usage per query", "tokens"
        )

        self.search_results_count = measure_module.MeasureInt(
            "search_results_count", "Number of search results returned", "results"
        )

        self.cache_hit_count = measure_module.MeasureInt(
            "cache_hit_count", "Cache hit count", "hits"
        )

        # Define views
        self._setup_views()

        # Custom event tracking
        self.events = []

    def _setup_views(self):
        """Set up metric views for aggregation"""

        # Query latency view
        query_latency_view = view_module.View(
            "query_latency_distribution",
            "Distribution of query latencies",
            [],
            self.query_latency_ms,
            aggregation_module.DistributionAggregation(
                [50, 100, 200, 500, 1000, 2000, 5000, 10000]
            ),
        )

        # Agent latency by agent type
        agent_latency_view = view_module.View(
            "agent_latency_by_type",
            "Agent latency by type",
            ["agent_type"],
            self.agent_latency_ms,
            aggregation_module.DistributionAggregation([10, 50, 100, 500, 1000, 2000]),
        )

        # Confidence score distribution
        confidence_view = view_module.View(
            "confidence_distribution",
            "Distribution of answer confidence scores",
            [],
            self.confidence_score,
            aggregation_module.DistributionAggregation(
                [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
            ),
        )

        # Token usage
        token_view = view_module.View(
            "token_usage",
            "Token usage per query",
            ["model"],
            self.token_count,
            aggregation_module.SumAggregation(),
        )

        # Search results
        search_results_view = view_module.View(
            "search_results_distribution",
            "Distribution of search result counts",
            [],
            self.search_results_count,
            aggregation_module.DistributionAggregation([0, 5, 10, 20, 50]),
        )

        # Cache metrics
        cache_hit_view = view_module.View(
            "cache_hit_rate",
            "Cache hit rate",
            ["cache_type"],
            self.cache_hit_count,
            aggregation_module.CountAggregation(),
        )

        # Register views
        views = [
            query_latency_view,
            agent_latency_view,
            confidence_view,
            token_view,
            search_results_view,
            cache_hit_view,
        ]

        for view in views:
            self.view_manager.register_view(view)

    def track_query_metrics(
        self,
        latency_ms: float,
        confidence: float,
        token_count: int,
        is_cached: bool = False,
    ):
        """Track metrics for a query"""
        mmap = self.stats.stats_recorder.new_measurement_map()

        # Record measurements
        mmap.measure_float_put(self.query_latency_ms, latency_ms)
        mmap.measure_float_put(self.confidence_score, confidence)
        mmap.measure_int_put(self.token_count, token_count)

        if is_cached:
            mmap.measure_int_put(self.cache_hit_count, 1)

        mmap.record()

        # Send to Application Insights
        self.telemetry_client.track_metric("query_latency_ms", latency_ms)
        self.telemetry_client.track_metric("confidence_score", confidence)
        self.telemetry_client.track_metric("token_count", token_count)

    def track_agent_metrics(self, agent_name: str, latency_ms: float):
        """Track metrics for individual agents"""
        mmap = self.stats.stats_recorder.new_measurement_map()
        tag_map = tag_map_module.TagMap()
        tag_map.insert("agent_type", agent_name)

        mmap.measure_float_put(self.agent_latency_ms, latency_ms)
        mmap.record(tag_map)

        # Custom event
        self.telemetry_client.track_event(
            "agent_execution",
            properties={"agent_name": agent_name, "latency_ms": str(latency_ms)},
        )

    def track_search_metrics(self, result_count: int, search_type: str = "hybrid"):
        """Track search metrics"""
        mmap = self.stats.stats_recorder.new_measurement_map()
        mmap.measure_int_put(self.search_results_count, result_count)
        mmap.record()

        self.telemetry_client.track_event(
            "search_executed",
            properties={"search_type": search_type, "result_count": str(result_count)},
        )

    def track_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Track errors with context"""
        properties = {
            "error_type": error_type,
            "error_message": error_message[:500],  # Limit message length
            "timestamp": datetime.utcnow().isoformat(),
        }

        if context:
            properties.update({k: str(v)[:100] for k, v in context.items()})

        self.telemetry_client.track_exception(
            type=error_type, value=error_message, properties=properties
        )

        logger.error(
            f"Error tracked: {error_type} - {error_message}",
            extra={"custom_dimensions": properties},
        )

    def track_custom_event(self, event_name: str, properties: Dict[str, Any]):
        """Track custom events"""
        self.telemetry_client.track_event(
            event_name, properties={k: str(v) for k, v in properties.items()}
        )

    def flush(self):
        """Flush all pending telemetry"""
        self.telemetry_client.flush()


class PerformanceMonitor:
    """Monitor and profile performance of the Q&A system"""

    def __init__(self, metrics_manager: MetricsManager):
        self.metrics = metrics_manager
        self.profiling_enabled = (
            os.getenv("ENABLE_PROFILING", "false").lower() == "true"
        )

    @asynccontextmanager
    async def track_operation(self, operation_name: str, **tags):
        """Context manager to track operation timing"""
        start_time = time.time()

        try:
            yield
        finally:
            elapsed_ms = (time.time() - start_time) * 1000

            # Track metric
            if "agent" in tags:
                self.metrics.track_agent_metrics(tags["agent"], elapsed_ms)

            # Log if slow
            if elapsed_ms > 1000:  # Log operations over 1 second
                logger.warning(
                    f"Slow operation: {operation_name} took {elapsed_ms:.2f}ms",
                    extra={"tags": tags},
                )

    def profile_async(self, func: Callable) -> Callable:
        """Decorator to profile async functions"""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.profiling_enabled:
                return await func(*args, **kwargs)

            start_time = time.time()
            start_memory = self._get_memory_usage()

            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                memory_delta = self._get_memory_usage() - start_memory

                self.metrics.track_custom_event(
                    "function_profile",
                    {
                        "function_name": func.__name__,
                        "elapsed_ms": elapsed_ms,
                        "memory_delta_mb": memory_delta / 1024 / 1024,
                    },
                )

                return result
            except Exception as e:
                self.metrics.track_error(
                    "function_error", str(e), {"function_name": func.__name__}
                )
                raise

        return wrapper

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss


class AutoGenAgentMonitor:
    """Specialized monitoring for AutoGen agents"""

    def __init__(self, metrics_manager: MetricsManager):
        self.metrics = metrics_manager
        self.agent_conversations = {}

    async def track_agent_conversation(
        self,
        agent_name: str,
        conversation_id: str,
        message: str,
        response: str,
        elapsed_ms: float,
    ):
        """Track AutoGen agent conversations"""

        # Store conversation
        if conversation_id not in self.agent_conversations:
            self.agent_conversations[conversation_id] = []

        self.agent_conversations[conversation_id].append(
            {
                "agent": agent_name,
                "timestamp": datetime.utcnow().isoformat(),
                "message_length": len(message),
                "response_length": len(response),
                "elapsed_ms": elapsed_ms,
            }
        )

        # Track metrics
        self.metrics.track_agent_metrics(agent_name, elapsed_ms)

        # Estimate tokens (rough approximation)
        estimated_tokens = (len(message) + len(response)) // 4
        self.metrics.track_query_metrics(
            latency_ms=elapsed_ms,
            confidence=1.0,  # Agent execution confidence
            token_count=estimated_tokens,
        )

        # Track slow agents
        if elapsed_ms > 2000:
            self.metrics.track_custom_event(
                "slow_agent_execution",
                {
                    "agent_name": agent_name,
                    "conversation_id": conversation_id,
                    "elapsed_ms": elapsed_ms,
                    "message_preview": message[:100],
                },
            )

    def get_agent_performance_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get performance summary for a conversation"""
        if conversation_id not in self.agent_conversations:
            return {}

        conversations = self.agent_conversations[conversation_id]

        # Calculate statistics
        agent_stats = {}
        for conv in conversations:
            agent = conv["agent"]
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "count": 0,
                    "total_ms": 0,
                    "avg_message_length": 0,
                    "avg_response_length": 0,
                }

            stats = agent_stats[agent]
            stats["count"] += 1
            stats["total_ms"] += conv["elapsed_ms"]
            stats["avg_message_length"] = (
                stats["avg_message_length"] * (stats["count"] - 1)
                + conv["message_length"]
            ) / stats["count"]
            stats["avg_response_length"] = (
                stats["avg_response_length"] * (stats["count"] - 1)
                + conv["response_length"]
            ) / stats["count"]

        # Calculate averages
        for agent, stats in agent_stats.items():
            stats["avg_ms"] = stats["total_ms"] / stats["count"]

        return {
            "conversation_id": conversation_id,
            "total_agents": len(agent_stats),
            "total_interactions": len(conversations),
            "agent_stats": agent_stats,
            "total_elapsed_ms": sum(c["elapsed_ms"] for c in conversations),
        }


class HealthChecker:
    """Health checking for all system components"""

    def __init__(self, orchestrator, data_store):
        self.orchestrator = orchestrator
        self.data_store = data_store
        self.checks = {
            "cosmos_db": self._check_cosmos_db,
            "search_service": self._check_search_service,
            "openai": self._check_openai,
            "blob_storage": self._check_blob_storage,
        }

    async def check_health(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }

        # Run all checks concurrently
        check_tasks = {name: check_func() for name, check_func in self.checks.items()}

        check_results = await asyncio.gather(
            *check_tasks.values(), return_exceptions=True
        )

        # Process results
        for (name, _), result in zip(check_tasks.items(), check_results):
            if isinstance(result, Exception):
                results["checks"][name] = {"status": "unhealthy", "error": str(result)}
                results["status"] = "degraded"
            else:
                results["checks"][name] = result
                if result["status"] != "healthy":
                    results["status"] = "degraded"

        return results

    async def _check_cosmos_db(self) -> Dict[str, Any]:
        """Check Cosmos DB connectivity"""
        try:
            start = time.time()
            # Simple query to check connectivity
            await asyncio.to_thread(
                lambda: list(
                    self.data_store.conversation_container.query_items(
                        query="SELECT VALUE COUNT(1) FROM c",
                        enable_cross_partition_query=True,
                        max_item_count=1,
                    )
                )
            )

            return {"status": "healthy", "latency_ms": (time.time() - start) * 1000}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_search_service(self) -> Dict[str, Any]:
        """Check Azure Cognitive Search"""
        try:
            start = time.time()
            # Simple search to check connectivity
            results = self.orchestrator.search_client.search(
                search_text="*", top=1, include_total_count=True
            )

            count = results.get_count()

            return {
                "status": "healthy",
                "latency_ms": (time.time() - start) * 1000,
                "document_count": count,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_openai(self) -> Dict[str, Any]:
        """Check Azure OpenAI"""
        try:
            start = time.time()
            # Simple embedding to check connectivity
            response = await self.orchestrator.aoai_client.embeddings.create(
                model=os.environ["AOAI_EMBED_DEPLOY"], input=["health check"]
            )

            return {
                "status": "healthy",
                "latency_ms": (time.time() - start) * 1000,
                "model": os.environ["AOAI_EMBED_DEPLOY"],
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_blob_storage(self) -> Dict[str, Any]:
        """Check Blob Storage"""
        try:
            start = time.time()
            # List containers to check connectivity
            containers = await asyncio.to_thread(
                lambda: list(
                    self.data_store.blob_service.list_containers(max_results=1)
                )
            )

            return {"status": "healthy", "latency_ms": (time.time() - start) * 1000}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Logging configuration for Azure
def setup_azure_logging():
    """Configure logging to send to Azure Application Insights"""
    instrumentation_key = os.getenv("APPLICATIONINSIGHTS_INSTRUMENTATION_KEY")

    if instrumentation_key:
        # Add Azure handler to root logger
        azure_handler = AzureLogHandler(
            connection_string=f"InstrumentationKey={instrumentation_key}"
        )

        # Configure handler
        azure_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        azure_handler.setFormatter(formatter)

        # Add custom properties
        azure_handler.add_telemetry_processor(add_custom_properties)

        # Add to root logger
        logging.getLogger().addHandler(azure_handler)


def add_custom_properties(envelope):
    """Add custom properties to all telemetry"""
    envelope.tags["ai.cloud.role"] = "confluence-qa-api"
    envelope.tags["ai.cloud.roleInstance"] = os.getenv("HOSTNAME", "unknown")

    # Add environment
    envelope.data.baseData.properties["environment"] = os.getenv(
        "ENVIRONMENT", "development"
    )
    envelope.data.baseData.properties["version"] = os.getenv("APP_VERSION", "1.0.0")

    return True


# Grafana Dashboard Configuration (as JSON)
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "Confluence Q&A System Monitoring",
        "panels": [
            {
                "title": "Query Latency",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, query_latency_ms)",
                        "legendFormat": "95th percentile",
                    }
                ],
            },
            {
                "title": "Agent Performance",
                "targets": [
                    {
                        "expr": "avg(agent_latency_ms) by (agent_type)",
                        "legendFormat": "{{agent_type}}",
                    }
                ],
            },
            {
                "title": "Confidence Scores",
                "targets": [
                    {
                        "expr": "avg(confidence_score)",
                        "legendFormat": "Average Confidence",
                    }
                ],
            },
            {
                "title": "Cache Hit Rate",
                "targets": [
                    {
                        "expr": "rate(cache_hit_count[5m])",
                        "legendFormat": "Cache Hit Rate",
                    }
                ],
            },
            {
                "title": "Token Usage",
                "targets": [
                    {
                        "expr": "sum(rate(token_count[5m])) by (model)",
                        "legendFormat": "{{model}}",
                    }
                ],
            },
            {
                "title": "Error Rate",
                "targets": [
                    {
                        "expr": "rate(error_count[5m])",
                        "legendFormat": "Errors per second",
                    }
                ],
            },
        ],
    }
}


# Example usage in the orchestrator
class MonitoredOrchestrator:
    """Wrapper to add monitoring to the orchestrator"""

    def __init__(self, orchestrator, metrics_manager: MetricsManager):
        self.orchestrator = orchestrator
        self.metrics = metrics_manager
        self.performance_monitor = PerformanceMonitor(metrics_manager)
        self.agent_monitor = AutoGenAgentMonitor(metrics_manager)

    @PerformanceMonitor.profile_async
    async def process_query(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Process query with full monitoring"""
        start_time = time.time()

        try:
            # Track query start
            self.metrics.track_custom_event(
                "query_started",
                {"conversation_id": conversation_id, "query_length": len(query)},
            )

            # Process with monitoring
            async with self.performance_monitor.track_operation(
                "process_query", conversation_id=conversation_id
            ):
                result = await self.orchestrator.process_query(query, conversation_id)

            # Track completion metrics
            elapsed_ms = (time.time() - start_time) * 1000
            confidence = result.get("confidence", 0)

            self.metrics.track_query_metrics(
                latency_ms=elapsed_ms,
                confidence=confidence,
                token_count=self._estimate_tokens(query, result.get("answer", "")),
                is_cached=False,
            )

            # Get agent performance summary
            agent_summary = self.agent_monitor.get_agent_performance_summary(
                conversation_id
            )

            # Add monitoring data to result
            result["monitoring"] = {
                "elapsed_ms": elapsed_ms,
                "agent_summary": agent_summary,
            }

            return result

        except Exception as e:
            # Track error
            self.metrics.track_error(
                "query_processing_error",
                str(e),
                {"conversation_id": conversation_id, "query": query[:100]},
            )
            raise

    def _estimate_tokens(self, query: str, answer: str) -> int:
        """Rough token estimation"""
        return (len(query) + len(answer)) // 4
