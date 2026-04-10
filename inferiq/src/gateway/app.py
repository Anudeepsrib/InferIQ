"""FastAPI inference gateway with lifespan management."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.backends.vllm_backend import VLLMBackend
from src.backends.nim_backend import NIMBackend
from src.config.settings import get_settings
from src.gateway.health import health_router, get_health_manager
from src.gateway.middleware import setup_middleware
from src.gateway.router import ModelRouter, RoutingStrategy
from src.gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    GenerateParams,
    ModelListResponse,
    ErrorResponse,
    ModelInfo,
)
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Global router instance
model_router: ModelRouter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager - load models on startup."""
    global model_router
    
    logger.info("Starting InferIQ Gateway...")
    
    # Load settings
    settings = get_settings()
    
    # Configure logging
    configure_logging(
        level=settings.benchmark.logging.level,
        format_type=settings.benchmark.logging.format,
        log_file=settings.benchmark.logging.file,
    )
    
    # Initialize router
    model_router = ModelRouter(strategy=RoutingStrategy.LEAST_LATENCY)
    health_manager = get_health_manager()
    
    # Load configured backends
    for model_config in settings.models:
        try:
            logger.info(
                "Loading model",
                name=model_config.name,
                backend=model_config.backend.value,
            )
            
            if model_config.backend.value == "vllm":
                backend = VLLMBackend(model_config)
            elif model_config.backend.value == "nim":
                backend = NIMBackend(model_config)
            elif model_config.backend.value == "nemo":
                from src.backends.nemo_backend import NeMoBackend
                backend = NeMoBackend(model_config)
                if not backend.available:
                    logger.warning(
                        "NeMo not available, skipping model",
                        name=model_config.name,
                    )
                    continue
            else:
                logger.warning(
                    "Unknown backend, skipping model",
                    name=model_config.name,
                    backend=model_config.backend.value,
                )
                continue
            
            # Load model
            await backend.load_model()
            
            # Register with router and health manager
            model_router.register_backend(backend, model_config)
            health_manager.register_backend(model_config.name, backend)
            
            logger.info("Model loaded successfully", name=model_config.name)
            
        except Exception as e:
            logger.error(
                "Failed to load model",
                name=model_config.name,
                error=str(e),
            )
    
    logger.info(
        "Gateway startup complete",
        loaded_models=len(model_router.backends) if model_router else 0,
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down InferIQ Gateway...")
    
    if model_router:
        for model_name, instances in model_router.backends.items():
            for instance in instances:
                try:
                    await instance.backend.shutdown()
                    logger.info("Backend shutdown", model=model_name)
                except Exception as e:
                    logger.error(
                        "Error during backend shutdown",
                        model=model_name,
                        error=str(e),
                    )
    
    logger.info("Gateway shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="InferIQ Gateway",
        description="Production-grade GPU-optimized LLM inference gateway",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Setup middleware
    setup_middleware(app, rate_limit=settings.gateway.rate_limit_requests_per_minute)
    
    # Include health router
    app.include_router(health_router)
    
    return app


app = create_app()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        error=str(exc),
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "type": "internal_error",
                "message": "An internal error occurred",
            },
        ).model_dump(),
    )


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List available models and their backends."""
    if model_router is None:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    
    models = model_router.get_model_list()
    
    return ModelListResponse(
        data=[
            ModelInfo(
                id=m.name,
                backend=m.backend,
                display_name=m.display_name,
                parameters=m.parameters,
                tags=m.tags,
            )
            for m in models
        ]
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    """OpenAI-compatible completions endpoint."""
    if model_router is None:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    
    # Convert to generate params
    params = GenerateParams(
        max_tokens=request.max_tokens or 16,
        temperature=request.temperature or 1.0,
        top_p=request.top_p or 1.0,
        stop_sequences=request.stop or [],
        seed=request.seed,
    )
    
    try:
        # Handle batch prompts
        if isinstance(request.prompt, list):
            results, instance = await model_router.route_generate_batch(
                request.model, request.prompt, params
            )
            
            choices = []
            for i, result in enumerate(results):
                choices.append(CompletionChoice(
                    text=result.text,
                    index=i,
                    finish_reason=result.finish_reason,
                ))
            
            total_usage = CompletionUsage(
                prompt_tokens=sum(r.prompt_tokens for r in results),
                completion_tokens=sum(r.completion_tokens for r in results),
                total_tokens=sum(r.total_tokens for r in results),
            )
        else:
            result, instance = await model_router.route_generate(
                request.model, request.prompt, params
            )
            
            choices = [CompletionChoice(
                text=result.text,
                index=0,
                finish_reason=result.finish_reason,
            )]
            
            total_usage = CompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            )
        
        return CompletionResponse(
            id=f"inferiq-{asyncio.get_event_loop().time()}",
            model=request.model,
            choices=choices,
            usage=total_usage,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Completion failed", model=request.model, error=str(e))
        raise HTTPException(status_code=500, detail="Inference failed")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""
    if model_router is None:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    
    # Concatenate messages into prompt (simplified - production would use chat template)
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    
    prompt = "\n".join(prompt_parts) + "\nAssistant:"
    
    # Convert to generate params
    params = GenerateParams(
        max_tokens=request.max_tokens or 16,
        temperature=request.temperature or 1.0,
        top_p=request.top_p or 1.0,
        stop_sequences=request.stop or [],
        seed=request.seed,
    )
    
    try:
        result, instance = await model_router.route_generate(
            request.model, prompt, params
        )
        
        return ChatCompletionResponse(
            id=f"inferiq-{asyncio.get_event_loop().time()}",
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=result.text,
                    ),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            ),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Chat completion failed", model=request.model, error=str(e))
        raise HTTPException(status_code=500, detail="Inference failed")


def main() -> None:
    """Entry point for running the gateway server."""
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.gateway.app:app",
        host=settings.gateway.host,
        port=settings.gateway.port,
        workers=settings.gateway.workers,
        reload=settings.gateway.reload,
        log_level=settings.gateway.log_level,
    )


if __name__ == "__main__":
    main()
