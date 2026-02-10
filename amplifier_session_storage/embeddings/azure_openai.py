"""
Azure OpenAI embedding provider implementation.

Supports text-embedding-3-large and other Azure OpenAI embedding models.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from azure.ai.inference.aio import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential

from .base import EmbeddingProvider
from .cache import EmbeddingCache

logger = logging.getLogger(__name__)


class AzureOpenAIEmbeddings(EmbeddingProvider):
    """
    Azure OpenAI embedding provider with caching.

    Supports models:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    """

    # Model dimension mappings
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        endpoint: str,
        model: str = "text-embedding-3-large",
        deployment: str | None = None,
        api_key: str | None = None,
        use_default_credential: bool = True,
        dimensions: int | None = None,
        cache_size: int = 1000,
    ):
        """
        Initialize Azure OpenAI embedding provider.

        Args:
            endpoint: Azure OpenAI endpoint URL (base URL or full deployment URL)
            model: Model name (default: text-embedding-3-large)
            deployment: Deployment name (defaults to model name if None)
            api_key: API key for authentication (optional if using RBAC)
            use_default_credential: Use Azure RBAC auth (default: True)
            dimensions: Vector dimensions (auto-detected if None)
            cache_size: Max cached embeddings (0 to disable)
        """
        self.endpoint = endpoint
        self.model = model
        self.deployment = deployment or model
        self.api_key = api_key
        self.use_default_credential = use_default_credential
        self._credential: DefaultAzureCredential | None = None

        # Auto-detect dimensions if not provided
        if dimensions is None:
            if model in self.MODEL_DIMENSIONS:
                self._dimensions = self.MODEL_DIMENSIONS[model]
            else:
                # Default to 3072 for unknown models (text-embedding-3-large)
                logger.warning(
                    f"Unknown model '{model}', defaulting to 3072 dimensions. "
                    f"Pass explicit dimensions parameter if different."
                )
                self._dimensions = 3072
        else:
            self._dimensions = dimensions

        self._client: EmbeddingsClient | None = None
        self._cache = EmbeddingCache(max_entries=cache_size) if cache_size > 0 else None

        auth_method = "RBAC" if use_default_credential else "API Key"
        logger.info(
            f"Azure OpenAI embeddings initialized: model={model}, deployment={self.deployment}, "
            f"dimensions={self._dimensions}, auth={auth_method}, cache_size={cache_size}"
        )

    @classmethod
    def from_env(cls) -> AzureOpenAIEmbeddings:
        """
        Create provider from environment variables.

        Required env vars:
            AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
                (can be base URL or full deployment URL with api-version)

        Optional env vars:
            AZURE_OPENAI_EMBEDDING_MODEL: Model name (default: text-embedding-3-large)
            AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Deployment name (defaults to model name)
            AZURE_OPENAI_API_KEY: API key (optional if using RBAC)
            AZURE_OPENAI_USE_RBAC: Use DefaultAzureCredential (default: true)
            AZURE_OPENAI_EMBEDDING_DIMENSIONS: Vector dimensions (auto-detected)
            AZURE_OPENAI_EMBEDDING_CACHE_SIZE: Cache size (default: 1000)

        Examples:
            # With RBAC (recommended)
            export AZURE_OPENAI_ENDPOINT="https://your-foundry.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15"
            export AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
            export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
            # Then: az login

            # With API key
            export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
            export AZURE_OPENAI_API_KEY="your-key"
            export AZURE_OPENAI_USE_RBAC="false"
        """
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        model = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        use_rbac_str = os.environ.get("AZURE_OPENAI_USE_RBAC", "true")
        dimensions_str = os.environ.get("AZURE_OPENAI_EMBEDDING_DIMENSIONS")
        cache_size_str = os.environ.get("AZURE_OPENAI_EMBEDDING_CACHE_SIZE", "1000")

        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable required")

        use_rbac = use_rbac_str.lower() in ("true", "1", "yes")

        # If using API key auth, require the key
        if not use_rbac and not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY required when AZURE_OPENAI_USE_RBAC=false")

        dimensions = int(dimensions_str) if dimensions_str else None
        cache_size = int(cache_size_str)

        return cls(
            endpoint=endpoint,
            model=model,
            deployment=deployment,
            api_key=api_key,
            use_default_credential=use_rbac,
            dimensions=dimensions,
            cache_size=cache_size,
        )

    @property
    def dimensions(self) -> int:
        """Number of dimensions in embedding vectors."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self.model

    async def _ensure_client(self) -> EmbeddingsClient:
        """Lazy initialize the Azure client with appropriate authentication."""
        if self._client is None:
            if self.use_default_credential:
                # Use Azure RBAC authentication
                # credential_scopes required for Azure OpenAI endpoints
                self._credential = DefaultAzureCredential()
                self._client = EmbeddingsClient(
                    endpoint=self.endpoint,
                    credential=self._credential,
                    credential_scopes=["https://cognitiveservices.azure.com/.default"],
                    model=self.deployment,  # Deployment name for routing
                )
                logger.info(
                    f"Using DefaultAzureCredential for Azure OpenAI (deployment={self.deployment})"
                )
            else:
                # Use API key authentication
                if not self.api_key:
                    raise ValueError("API key required when not using default credential")
                self._client = EmbeddingsClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.api_key),
                    model=self.deployment,  # Deployment name for routing
                )
                logger.info(f"Using API key for Azure OpenAI (deployment={self.deployment})")
        return self._client

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first
        if self._cache:
            cached = self._cache.get(text, self.model)
            if cached is not None:
                logger.debug(f"Cache hit for text (len={len(text)})")
                return cached

        # Generate embedding
        client = await self._ensure_client()

        # Note: For Foundry endpoints, the model/deployment is in the URL
        # The 'model' parameter in the API call is optional
        response = await client.embed(
            input=[text],
            dimensions=self._dimensions,
        )

        # Extract embedding and ensure it's a list[float]
        embedding_raw = response.data[0].embedding
        # Convert to list[float] regardless of input type
        if isinstance(embedding_raw, list):
            embedding = [float(x) for x in embedding_raw]
        else:
            embedding = [float(x) for x in embedding_raw]

        # Cache the result
        if self._cache:
            self._cache.put(text, self.model, embedding)
            logger.debug(f"Cached embedding for text (len={len(text)})")

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with partial caching.

        Checks cache for each text individually, only requests embeddings
        for cache misses, then caches new results.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        # Check cache for each text
        if self._cache:
            for i, text in enumerate(texts):
                cached = self._cache.get(text, self.model)
                if cached is not None:
                    results[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))

        # Generate embeddings for cache misses
        if texts_to_embed:
            client = await self._ensure_client()

            # Extract just the texts for API call
            texts_for_api = [text for _, text in texts_to_embed]

            # Note: For Foundry endpoints, the model/deployment is in the URL
            response = await client.embed(
                input=texts_for_api,
                dimensions=self._dimensions,
            )

            # Store results and cache
            for (original_idx, text), embedding_data in zip(
                texts_to_embed, response.data, strict=True
            ):
                # Extract and ensure list[float] type
                embedding_raw = embedding_data.embedding
                # Convert to list[float] regardless of input type
                if isinstance(embedding_raw, list):
                    embedding = [float(x) for x in embedding_raw]
                else:
                    embedding = [float(x) for x in embedding_raw]
                results[original_idx] = embedding

                if self._cache:
                    self._cache.put(text, self.model, embedding)

            logger.info(
                f"Generated {len(texts_to_embed)} embeddings, "
                f"{len(texts) - len(texts_to_embed)} from cache"
            )

        # Verify all results are populated (explicit check — assert is stripped with -O)
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise ValueError(
                f"Embedding generation incomplete: {len(missing)}/{len(results)} "
                f"results missing at indices {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        return results  # type: ignore

    async def close(self) -> None:
        """Close the Azure client and credential."""
        if self._client:
            await self._client.close()
            self._client = None

        if self._credential:
            await self._credential.close()
            self._credential = None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats()
        return {"cache_enabled": False}
