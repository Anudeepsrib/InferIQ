# InferIQ Security Considerations

## Model Endpoint Exposure
- The FastAPI gateway exposes OpenAI-compatible inference endpoints. In production, **never** expose directly to the public internet without authentication, rate limiting, and WAF.
- Use `INFERIQ_API_KEY` (implement in middleware) or OAuth2 / OIDC proxy in front of the gateway for non-local deployments.
- Default CORS is permissive in dev; production must set explicit `INFERIQ_CORS_ORIGINS`.

## Prompt / Completion Logging
- By default, `LoggingMiddleware` does **not** log full request/response bodies (see `log_request_body=False`).
- When enabled for debugging, prompts and completions can contain PII or sensitive data. Redact or disable in production.
- Authorization headers are redacted in logs.

## Local Benchmark Data & Artifacts
- Benchmark results, Chrome traces, NSight reports, and MLflow artifacts are written to `results/`, `traces/`, `profiling/`.
- These can contain model outputs, performance profiles, and indirectly training data characteristics. Treat as sensitive.
- `.gitignore` excludes them; never commit real benchmark outputs or traces.

## GPU Container Permissions
- Running vLLM / NeMo / NIM containers or the gateway with GPU access requires `--gpus all` or NVIDIA Container Toolkit.
- The non-root `appuser` in Dockerfiles may need additional capabilities or device permissions for NVML / CUDA in some environments.
- Never run containers as root in production.

## NIM / NGC Credentials
- NVIDIA NIM backends require NGC API keys for image pull and runtime (when using official NIM containers).
- **Never** hardcode NGC keys. Use Kubernetes secrets, Docker secrets, or injected env vars only.
- The `nim_backend.py` reads `api_key` from model config or `NIM_API_KEY` env; rotate regularly.

## Dependency Risks
- Heavy optional deps (vLLM, NeMo) have large attack surface and complex build requirements. Pin versions and use SBOM / `pip-audit` in CI.
- Base dependencies include torch (CPU wheels are safer for dev images).

## Production Hardening Checklist (not fully implemented)
- [ ] Mandatory API key / mTLS auth
- [ ] Request size & token limits enforced in middleware
- [ ] Full prompt redaction + audit logging to separate system
- [ ] Validated Kubernetes NetworkPolicy + PodSecurity
- [ ] Signed container images + SBOM
- [ ] Secret rotation for any backend credentials
- [ ] Chaos / failure injection tested for routing fallbacks
- [ ] Real A100/H100 benchmark data published with reproducible methodology (current dry-run is synthetic)

Report security issues via GitHub Issues (private) or email to maintainers.
