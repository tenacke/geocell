.PHONY: help test build clean install dev-install lint format h3lib clean-h3lib

# Default target
help:
	@echo "GEOCELL Build & Test Commands"
	@echo "=============================="
	@echo "make h3lib         - Build H3 C library (required first!)"
	@echo "make dev-install   - Install Python dev dependencies"
	@echo "make build         - Build Cython extensions (needs h3lib)"
	@echo "make test          - Build & run all tests"
	@echo "make test-only     - Run tests without building"
	@echo "make test-fast     - Run tests excluding slow ones"
	@echo "make install       - Full install (h3lib + build + test)"
	@echo "make clean         - Clean build artifacts"
	@echo "make clean-all     - Clean everything including H3"
	@echo "make lint          - Run code quality checks"
	@echo "make format        - Format code"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

# Build H3 C library from submodule
h3lib:
	@echo "Building H3 C library..."
	@if [ ! -d "src/h3lib/build" ]; then mkdir -p src/h3lib/build; fi
	@cd src/h3lib/build && \
		cmake -DCMAKE_BUILD_TYPE=Release .. && \
		make
	@echo "H3 library built successfully!"

# Clean H3 build
clean-h3lib:
	@echo "Cleaning H3 build..."
	rm -rf src/h3lib/build
	@echo "H3 clean complete!"

# Build Cython extensions (requires h3lib)
build:
	@echo "Checking H3 library..."
	@if [ ! -f "src/h3lib/build/lib/libh3.a" ]; then \
		echo "H3 library not found. Building it first..."; \
		$(MAKE) h3lib; \
	fi
	@echo "Building with scikit-build-core..."
	pip install -e ".[test]" -v
	@echo "Build complete!"

# Build without running tests (use with caution)
build-force:
	@echo "Checking H3 library..."
	@if [ ! -f "src/h3lib/build/lib/libh3.a" ]; then \
		echo "H3 library not found. Building it first..."; \
		$(MAKE) h3lib; \
	fi
	@echo "Building Cython extensions without tests..."
	pip install -e . --no-build-isolation
	@echo "Build complete!"

# Run tests (requires build first)
test: build
	@echo "Running tests..."
	pytest tests/ -v

# Run fast tests only (requires build first)
test-fast: build
	@echo "Running fast tests..."
	pytest tests/ -v -m "not slow"

# Run specific test file (requires build first)
test-%: build
	@echo "Running tests in tests/$*..."
	pytest tests/$* -v

# Run tests without rebuilding (use if already built)
test-only:
	@echo "Running tests (assuming already built)..."
	pytest tests/ -v

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.so" -delete
	find . -name "*.cpp" -path "*/geocell/_cython/*" -delete
	@echo "Clean complete!"
	rm -rf src/*.egg-info/
	rm -rf src/geocell/*.egg-info/
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.so' -delete
	find . -type f -name '*.c' -path '*/geocell/*' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Clean everything including H3
clean-all: clean clean-h3lib
	@echo "Full clean complete!"

# Install the package (full build with H3, then test)
install: h3lib dev-install build test-only
	@echo "Full installation complete!"

# Install in development mode (Python dependencies only)
dev-install:
	@echo "Installing development dependencies..."
	pip install -e ".[dev]"
	@echo "Development setup complete!"

# Code quality checks (future)
lint:
	@echo "Linting not yet configured"

# Code formatting (future)
format:
	@echo "Formatting not yet configured"

# Coverage report
coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=geocell --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# Quick sanity check
check: test-fast build-force
	@echo "Quick check complete!"
