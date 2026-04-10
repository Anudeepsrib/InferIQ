"""Tests for FastAPI inference gateway."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.gateway.app import app, model_router
from src.gateway.health import get_health_manager
from src.gateway.router import ModelRouter, RoutingStrategy
from src.gateway.schemas import (
    CompletionRequest,
    ChatCompletionRequest,
    ChatMessage,
    GenerateResult,
    ModelBackend,
    ModelConfig,
)


# Fixtures
@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model_config():
    """Create mock model config."""
    return ModelConfig(
        name="test-model",
        display_name="Test Model",
        model_id="test/model",
        backend=ModelBackend.VLLM,
        parameters={"size": "7B"},
        config={},
        tags=["test"],
    )


@pytest.fixture
def mock_generate_result():
    """Create mock generation result."""
    return GenerateResult(
        text="Generated text",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        ttft_ms=50.0,
        total_time_ms=100.0,
        tokens_per_second=200.0,
        finish_reason="stop",
    )


# Health endpoint tests
class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check_no_backends(self, client):
        """Test health check with no backends registered."""
        response = client.get("/health")
        # Should be healthy even with no backends (gateway itself is running)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_readiness_check_no_backends(self, client):
        """Test readiness check with no backends."""
        response = client.get("/ready")
        # Should fail readiness if no backends are loaded
        assert response.status_code == 503
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "inferiq_requests_total" in response.text or "# HELP" in response.text


# Model list endpoint tests
class TestModelEndpoints:
    """Test model listing endpoints."""
    
    def test_list_models_no_backends(self, client):
        """Test model list with no backends."""
        response = client.get("/v1/models")
        assert response.status_code == 503


# Completion endpoint tests
class TestCompletionEndpoints:
    """Test completion endpoints."""
    
    def test_completions_no_backends(self, client):
        """Test completions with no backends."""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            max_tokens=10,
        )
        response = client.post("/v1/completions", json=request.model_dump())
        # Should fail with 503 if no backends available
        assert response.status_code == 503
    
    def test_chat_completions_no_backends(self, client):
        """Test chat completions with no backends."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=10,
        )
        response = client.post("/v1/chat/completions", json=request.model_dump())
        assert response.status_code == 503


# Router tests
class TestModelRouter:
    """Test model routing logic."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        router = ModelRouter(strategy=RoutingStrategy.ROUND_ROBIN)
        assert router.strategy == RoutingStrategy.ROUND_ROBIN
        assert len(router.backends) == 0
    
    def test_router_round_robin(self):
        """Test round-robin routing strategy."""
        router = ModelRouter(strategy=RoutingStrategy.ROUND_ROBIN)
        
        # Mock backends would need to be registered here
        # This test would need actual backend mocks
        pass
    
    def test_router_least_latency(self):
        """Test least latency routing strategy."""
        router = ModelRouter(strategy=RoutingStrategy.LEAST_LATENCY)
        assert router.strategy == RoutingStrategy.LEAST_LATENCY
    
    def test_router_least_loaded(self):
        """Test least loaded routing strategy."""
        router = ModelRouter(strategy=RoutingStrategy.LEAST_LOADED)
        assert router.strategy == RoutingStrategy.LEAST_LOADED


# Request/Response schema tests
class TestSchemas:
    """Test Pydantic schema validation."""
    
    def test_completion_request_validation(self):
        """Test completion request validation."""
        # Valid request
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            max_tokens=100,
            temperature=0.7,
        )
        assert request.model == "test-model"
        assert request.max_tokens == 100
        
        # Test stop sequence normalization
        request_with_string_stop = CompletionRequest(
            model="test-model",
            prompt="Hello",
            stop="end",
        )
        assert request_with_string_stop.stop == ["end"]
        
        request_with_list_stop = CompletionRequest(
            model="test-model",
            prompt="Hello",
            stop=["end", "stop"],
        )
        assert request_with_list_stop.stop == ["end", "stop"]
    
    def test_chat_completion_request_validation(self):
        """Test chat completion request validation."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Hello"),
            ],
            max_tokens=100,
        )
        assert len(request.messages) == 2
    
    def test_generate_result_creation(self):
        """Test GenerateResult creation."""
        result = GenerateResult(
            text="Generated",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            ttft_ms=50.0,
            total_time_ms=100.0,
            tokens_per_second=200.0,
            finish_reason="stop",
        )
        assert result.text == "Generated"
        assert result.total_tokens == 30


# Middleware tests
class TestMiddleware:
    """Test middleware functionality."""
    
    def test_request_id_header(self, client):
        """Test that request ID header is present."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
    
    def test_process_time_header(self, client):
        """Test that process time header is present."""
        response = client.get("/health")
        assert "X-Process-Time-Ms" in response.headers


# Error handling tests
class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_model(self, client):
        """Test request for non-existent model."""
        request = CompletionRequest(
            model="non-existent-model",
            prompt="Hello",
        )
        response = client.post("/v1/completions", json=request.model_dump())
        # Should get 404 or 503 depending on router state
        assert response.status_code in [404, 503]
    
    def test_invalid_request_body(self, client):
        """Test invalid request body."""
        response = client.post("/v1/completions", json={"invalid": "body"})
        # Should get 422 validation error
        assert response.status_code == 422
