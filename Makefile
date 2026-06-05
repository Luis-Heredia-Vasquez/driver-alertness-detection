PY=python
PYTHON_FILES=$(shell find src scripts tests -name "*.py" -type f)

.PHONY: help install train evaluate webcam serve test lint format clean docker-build docker-up

help:
	@echo "Driver Alertness Detection System - Available Targets"
	@echo "====================================================="
	@echo "Installation & Setup:"
	@echo "  make install           - Install dependencies"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  make train             - Train model with default config"
	@echo "  make train-resume      - Resume training from checkpoint"
	@echo "  make evaluate          - Evaluate best model"
	@echo ""
	@echo "Inference & Demos:"
	@echo "  make webcam            - Run webcam demo"
	@echo "  make serve             - Run Flask API server"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test              - Run pytest suite"
	@echo "  make test-cov          - Run tests with coverage report"
	@echo "  make lint              - Run pylint checks"
	@echo "  make format            - Format code with black/isort"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build      - Build Docker image"
	@echo "  make docker-up         - Run Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove caches and temp files"
	@echo "  make clean-models      - Remove trained models"
	@echo ""

install:
	$(PY) -m pip install -r requirements.txt
	@echo "✓ Dependencies installed"

train:
	$(PY) scripts/train.py --config configs/default.yaml --data-dir data/ --output-dir outputs/

train-resume:
	@if [ -f "outputs/models/best_model.pt" ]; then \
		$(PY) scripts/train.py --resume outputs/models/best_model.pt; \
	else \
		echo "Error: No checkpoint found at outputs/models/best_model.pt"; \
		exit 1; \
	fi

evaluate:
	@if [ -f "outputs/models/best_model.pt" ]; then \
		$(PY) scripts/evaluate.py --checkpoint outputs/models/best_model.pt --output-dir outputs/plots/; \
	else \
		echo "Error: No checkpoint found at outputs/models/best_model.pt. Run 'make train' first."; \
		exit 1; \
	fi

webcam:
	$(PY) scripts/run_webcam.py --config configs/default.yaml

serve:
	$(PY) -m flask --app src.api.app run --host=0.0.0.0 --port=5000

test:
	pytest -q

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term-missing tests/

lint:
	pylint $(PYTHON_FILES) --disable=all --enable=E,F

format:
	isort $(PYTHON_FILES)
	black $(PYTHON_FILES)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned up cache and temp files"

clean-models:
	rm -rf outputs/models/*.pt outputs/models/*.pth
	@echo "✓ Removed trained models"

docker-build:
	docker build -t driver-alertness:latest .
	@echo "✓ Docker image built"

docker-up:
	docker-compose up --build

.DEFAULT_GOAL := help
