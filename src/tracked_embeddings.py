"""
Proper embedding usage tracking via HTTP response interception.

This module provides LangSmith tracking for OpenAI embedding operations by
intercepting HTTP responses before LangChain processes them. This captures
actual token counts from the API instead of using estimation heuristics.

Key Components:
- UsageCapturingHTTPClient: Custom httpx.Client that extracts usage metadata
- TrackedOpenAIEmbeddings: OpenAIEmbeddings subclass with LangSmith integration
"""

import threading
import os
import time  # TEMPORARY DEBUG: for timing instrumentation
import logging  # TEMPORARY DEBUG: for timing instrumentation
from typing import Dict, List, Any, ClassVar, Optional
import httpx
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree


class UsageCapturingHTTPClient(httpx.Client):
    """
    HTTP client that captures OpenAI API usage metadata from responses.

    This client intercepts responses from the OpenAI embeddings API before
    LangChain processes them, extracting the 'usage' field that contains
    actual token counts. Usage is accumulated across multiple requests to
    handle batching automatically.

    Thread-safe for concurrent embedding operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._usage_data = {"prompt_tokens": 0, "total_tokens": 0}
        self._lock = threading.Lock()

    def request(self, method: str, url: Any, **kwargs) -> httpx.Response:
        """
        Intercept HTTP request to capture usage metadata from response.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional request parameters

        Returns:
            HTTP response object
        """
        response = super().request(method, url, **kwargs)

        # Only process embeddings endpoints with successful responses
        if "/embeddings" in str(url) and response.status_code == 200:
            self._extract_usage(response)

        return response

    def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        """
        Intercept send() method which is used by the OpenAI SDK.

        Args:
            request: The HTTP request to send
            **kwargs: Additional send parameters

        Returns:
            HTTP response object
        """
        response = super().send(request, **kwargs)

        # Only process embeddings endpoints with successful responses
        if "/embeddings" in str(request.url) and response.status_code == 200:
            self._extract_usage(response)

        return response

    def _extract_usage(self, response: httpx.Response) -> None:
        """
        Extract and accumulate usage metadata from API response.

        OpenAI embeddings API returns responses in the format:
        {
            "data": [{"embedding": [...]}],
            "usage": {"prompt_tokens": N, "total_tokens": N}
        }

        LangChain only extracts the embeddings, discarding usage data.
        This method captures it before it's lost.

        IMPORTANT: We read the raw content without consuming the response stream,
        so the OpenAI SDK can still parse it normally.

        Args:
            response: HTTP response from embeddings API
        """
        try:
            # Read the response content without consuming the stream
            # Use response.content which httpx caches after reading
            import json
            content = response.content
            data = json.loads(content)

            if "usage" in data:
                with self._lock:
                    self._usage_data["prompt_tokens"] += data["usage"].get("prompt_tokens", 0)
                    self._usage_data["total_tokens"] += data["usage"].get("total_tokens", 0)
        except Exception:
            # Silent fail - don't break embeddings on parsing errors
            pass

    def get_and_reset_usage(self) -> Dict[str, int]:
        """
        Get accumulated usage data and reset counters.

        This method is called after embedding operations complete to retrieve
        the total usage across all API calls (including batches). Counters are
        reset to prevent double-counting on subsequent operations.

        Returns:
            Dictionary with 'prompt_tokens' and 'total_tokens' counts
        """
        with self._lock:
            usage = self._usage_data.copy()
            self._usage_data = {"prompt_tokens": 0, "total_tokens": 0}
            return usage


class TrackedOpenAIEmbeddings(OpenAIEmbeddings):
    """
    OpenAIEmbeddings with proper LangSmith usage tracking.

    This class extends OpenAIEmbeddings to capture actual token counts from
    API responses and report them to LangSmith. It uses HTTP response
    interception to extract usage metadata that LangChain normally discards.

    Features:
    - Actual token counts (not estimates)
    - Automatic batch handling
    - Thread-safe concurrent operations
    - LangSmith integration via usage_metadata
    - Works with OpenRouter and native OpenAI API

    Usage:
        embeddings = TrackedOpenAIEmbeddings(
            model="qwen/qwen3-embedding-8b",
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Automatic tracking in LangSmith
        vectors = embeddings.embed_query("hello world")
    """

    # Pricing constant for qwen/qwen3-embedding-8b on OpenRouter
    COST_PER_1M_TOKENS: ClassVar[float] = 0.10  # $0.10 per 1M tokens

    def __init__(self, **kwargs):
        """
        Initialize TrackedOpenAIEmbeddings with custom HTTP client.

        Args:
            **kwargs: All standard OpenAIEmbeddings parameters
                     (model, base_url, api_key, etc.)
        """
        # Create custom HTTP client for usage capture
        usage_client = UsageCapturingHTTPClient(
            timeout=kwargs.get("timeout", 60.0),
            headers=kwargs.get("default_headers"),
        )

        # Inject custom client into parent class
        kwargs["http_client"] = usage_client
        super().__init__(**kwargs)

        # Store reference to usage client after parent initialization
        # Use object.__setattr__ to bypass Pydantic's __setattr__
        object.__setattr__(self, '_usage_client', usage_client)

    @traceable(
        run_type="embedding",
        name="Embed Query",
        metadata={"ls_provider": "openrouter", "ls_model_name": "qwen/qwen3-embedding-8b"}
    )
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query with usage tracking.

        Makes direct API call with our custom HTTP client to capture usage,
        then reports to LangSmith within the trace context.

        Args:
            text: Query text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # TEMPORARY DEBUG: Start timing
        start = time.perf_counter()

        # Make direct API call using parent's method
        # This ensures we stay within the @traceable context
        result = super().embed_query(text)

        # TEMPORARY DEBUG: Log timing
        elapsed = time.perf_counter() - start
        logging.info(f"⚡ Embedding API call: {elapsed:.3f}s")

        # Report usage while still in traced context
        # Pass the result so we can properly structure outputs
        self._report_usage(result)

        return result

    @traceable(
        run_type="embedding",
        name="Embed Documents",
        metadata={"ls_provider": "openrouter", "ls_model_name": "qwen/qwen3-embedding-8b"}
    )
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with usage tracking.

        Makes direct API call with our custom HTTP client to capture usage,
        then reports to LangSmith within the trace context.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors (one per document)
        """
        # TEMPORARY DEBUG: Start timing
        start = time.perf_counter()

        # Make direct API call using parent's method
        # This ensures we stay within the @traceable context
        result = super().embed_documents(texts)

        # TEMPORARY DEBUG: Log timing
        elapsed = time.perf_counter() - start
        logging.info(f"⚡ Embedding API batch call ({len(texts)} docs): {elapsed:.3f}s")

        # Report usage while still in traced context
        # Pass the result so we can properly structure outputs
        self._report_usage(result)

        return result

    def _report_usage(self, embeddings: Any) -> None:
        """
        Retrieve usage from HTTP client and report to LangSmith.

        This method:
        1. Gets accumulated usage from the custom HTTP client
        2. Calculates cost based on actual token counts
        3. Uses run.end() to properly set outputs with usage_metadata
           This ensures tokens appear in both metadata AND run overview

        Args:
            embeddings: The embedding result (List[float] or List[List[float]])

        If not in a traced context (LangSmith disabled), fails silently.
        """
        usage = self._usage_client.get_and_reset_usage()

        if usage["total_tokens"] > 0:
            # Calculate actual cost based on real token counts
            cost = usage["prompt_tokens"] * self.COST_PER_1M_TOKENS / 1_000_000

            usage_metadata = {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": 0,  # Embeddings don't have output tokens
                "total_tokens": usage["total_tokens"],
                "input_cost": cost,
                "output_cost": 0.0,
                "total_cost": cost,
            }

            try:
                run = get_current_run_tree()
                if run:
                    # Set usage_metadata in metadata section
                    run.set(usage_metadata=usage_metadata)

                    # End the run with outputs that include usage_metadata
                    # This makes tokens appear in run overview
                    run.end(outputs={
                        "embeddings": embeddings,
                        "usage_metadata": usage_metadata
                    })
            except Exception:
                # Silent fail if not in traced context
                # This allows the embeddings to work even when LangSmith is disabled
                pass
