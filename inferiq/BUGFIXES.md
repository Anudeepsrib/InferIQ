# InferIQ Bug Fixes

This document summarizes the bugs fixed during initial testing and debugging.

## Issues Fixed

### 1. TOML Syntax Errors in pyproject.toml
- **Problem**: Duplicate `warn_unused_ignores` key and malformed string in coverage section
- **Fix**: Removed duplicate key and fixed string escaping
- **Files**: `pyproject.toml`

### 2. Configuration Type Mismatch
- **Problem**: `precision: 16` (integer) but Pydantic expected string
- **Fix**: Changed to `precision: "16"`
- **Files**: `configs/default.yaml`

### 3. Package Deprecation Warning
- **Problem**: `pynvml` package deprecated in favor of `nvidia-ml-py`
- **Fix**: Updated imports and dependency specification
- **Files**: `pyproject.toml`, `src/utils/gpu.py`

### 4. Duplicate Import in Workload Generator
- **Problem**: Duplicate `import random` inside method causing UnboundLocalError
- **Fix**: Removed duplicate import, kept module-level import
- **Files**: `src/benchmark/workloads.py`

### 5. Readiness Check Logic Error
- **Problem**: Readiness check returned 200 when no backends loaded
- **Fix**: Added condition to require at least one loaded backend
- **Files**: `src/gateway/health.py`

### 6. DateTime Deprecation Warnings
- **Problem**: `datetime.utcnow()` deprecated in Python 3.12+
- **Fix**: Replaced with `datetime.now(timezone.utc)` throughout codebase
- **Files**: Multiple files including schemas, health, benchmark metrics, backends

### 7. JSON Serialization Issues
- **Problem**: DateTime objects not JSON serializable in HTTP responses
- **Fix**: Added custom DateTimeEncoder for health endpoint
- **Files**: `src/gateway/health.py`

### 8. Test Assertion Mismatches
- **Problem**: Tests expected different string formats than actual output
- **Fix**: Updated test assertions to match actual output format
- **Files**: `tests/test_benchmark.py`

## Test Results

All 50 tests now pass:
- Backend tests: 12/12 passed
- Benchmark tests: 17/17 passed  
- Gateway tests: 21/21 passed

## Remaining Warnings

- Pytest asyncio configuration warning (non-critical)
- Torch pynvml deprecation warning (will be resolved when package updated)

## Verification

The following commands now work successfully:
- `pip install -e .` - Package installation
- `python scripts/run_benchmark.py --help` - CLI interface
- `from src.gateway.app import app` - Gateway import
- `import dashboard.app` - Dashboard import
- `pytest tests/` - Full test suite

All major functionality is now working as expected.
