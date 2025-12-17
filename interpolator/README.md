# 5D Neural Network Interpolator

A full-stack research-grade system for learning and serving neural network models that can interpolate any 5-dimensional numerical dataset. The system provides an intuitive web interface for uploading datasets, training custom neural networks, and making predictions.

## Overview

This project implements a complete machine learning pipeline:
- **Backend**: FastAPI server with neural network training and prediction endpoints
- **Frontend**: Next.js web application with an interactive UI for dataset management and model training
- **Neural Network**: Lightweight TensorFlow/Keras implementation optimized for 5D regression tasks (CPU-only mode)

The system automatically handles data preprocessing (missing values, standardization), train/validation/test splits, and provides comprehensive metrics and visualizations.

**Note**: TensorFlow is configured to use CPU-only mode, ensuring consistent performance across different hardware configurations and avoiding GPU dependencies.

**📦 [Install from PyPI](https://pypi.org/project/pydis-nn/)** | **📚 [Full Documentation](https://pydis-nn.readthedocs.io/)** | **📊 [Performance Profiling](https://pydis-nn.readthedocs.io/en/latest/performance.html)**

## Quick Start

Get up and running in minutes:

```bash
# Clone the repository
git clone <repository-url>
cd interpolator

# Start the application
./scripts/launch-stack.sh

# Open your browser to http://localhost:3000
```

That's it! The application will be running with:
- Frontend UI at http://localhost:3000
- Backend API at http://localhost:8000
- API documentation at http://localhost:8000/docs

See the [Local Development Setup](#local-development-setup) section below for detailed installation and configuration options.

## Prerequisites

- **Python** 3.10+ (for backend)
- **Node.js** 20+ (for frontend)
- **Docker & Docker Compose** (optional, recommended for production)

## Environment Variables

### Backend

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PYTHONPATH` | Python module search path | `/app` | Yes |
| `PYTHONUNBUFFERED` | Disable Python output buffering | `1` | No |

### Frontend

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NODE_ENV` | Node.js environment | `production` | Yes |
| `NEXT_PUBLIC_API_URL` | Backend API base URL | `http://localhost:8000` | Yes |
| `NEXT_TELEMETRY_DISABLED` | Disable Next.js telemetry | `1` | No |

## Local Development Setup

**Note:** All commands should be run from the `interpolator/` directory. First navigate to the project directory:

```bash
cd interpolator
```

### Option 1: Docker Compose (Recommended)

#### Using Convenience Scripts (Recommended)

Start both backend and frontend services with health checks:

```bash
./scripts/launch-stack.sh
```

Stop services:

```bash
./scripts/docker-stop.sh
```

View logs:

```bash
./scripts/docker-logs.sh
```

#### Using Raw Docker Compose Commands

Alternatively, you can use Docker Compose directly:

Start services:

```bash
docker-compose up --build
```

Stop services:

```bash
docker-compose down
```

View logs:

```bash
./scripts/docker-logs.sh
```

Or directly:

```bash
docker-compose logs -f
```

**Note**: The application includes comprehensive logging for debugging. When training models, you'll see detailed logs showing:
- Dataset loading and preprocessing progress
- Model creation and training progress
- Training completion with timing information
- Any errors or warnings

View backend logs specifically to see training progress:

```bash
docker-compose logs -f backend
```

#### Service URLs

Services will be available at:
- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:3000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc

### Option 2: Manual Setup

#### Backend

1. Create and activate virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install package in editable mode:
```bash
pip install -e .
```

3. Start development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start development server:
```bash
npm run dev
```

Frontend will be available at http://localhost:3000

## Running Tests

### Full Pipeline Testing

To test the complete application pipeline (backend and frontend together):

```bash
cd interpolator
./scripts/test-pipeline.sh
```

This script:
1. Starts both backend and frontend services using Docker Compose
2. Waits for services to become healthy
3. Tests API endpoints (health check, root, API docs)
4. Provides status summary and next steps

This is useful for:
- Verifying the complete stack works end-to-end
- Testing after deployment
- Validating Docker Compose configuration

### Backend Tests

1. Install dev dependencies:
```bash
cd backend
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest tests/ -v
```

3. Run tests with coverage:
```bash
pytest tests/ -v --cov=pydis_nn --cov-report=html
```

4. View coverage report:
```bash
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Test Coverage

- Run with terminal output: `pytest --cov=pydis_nn --cov-report=term-missing`
- Generate HTML report: `pytest --cov=pydis_nn --cov-report=html`
- View HTML report: `open htmlcov/index.html`

Current test coverage includes:
- Data loading and preprocessing (17 tests)
- Neural network training and evaluation (13 tests)
- API endpoints (12 tests)

## Dataset Format

The system expects datasets in `.pkl` (pickle) format with the following structure:

```python
{
    'X': numpy.ndarray,  # Shape: (n_samples, 5) - 5 feature columns
    'y': numpy.ndarray   # Shape: (n_samples,) - target values
}
```

**Requirements:**
- Exactly 5 features (columns) in X
- X and y must have the same number of samples
- Missing values (NaN/inf) are automatically handled

**Example:**
```python
import pickle
import numpy as np

# Generate sample dataset
from pydis_nn.utils import generate_sample_dataset
data = generate_sample_dataset(n=1000, seed=42)

# Save to file
with open('my_dataset.pkl', 'wb') as f:
    pickle.dump(data, f)
```

## API Documentation

Interactive API documentation is available when the backend is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

- `GET /health` - Health check
- `POST /upload` - Upload a `.pkl` dataset file
- `POST /train` - Train a neural network model
- `POST /predict` - Make predictions with trained model

## Docker Build Details

### Backend

- **Base Image**: `python:3.12-slim`
- **Multi-stage Build**: 
  - `development`: Includes all dependencies, enables hot-reload
  - `production`: Optimized with non-root user for security
- **Health Check**: `/health` endpoint
- **Port**: 8000

### Frontend

- **Base Image**: `node:20-alpine`
- **Multi-stage Build**:
  - `deps`: Install dependencies
  - `builder`: Build Next.js application
  - `runner`: Production runtime (minimal image)
- **Standalone Build**: Enabled for smaller image size
- **Health Check**: Root endpoint
- **Port**: 3000

## Logging and Debugging

The application includes comprehensive logging to help debug issues. All logs are output to Docker logs and can be viewed using:

```bash
./scripts/docker-logs.sh
```

### Log Messages

The logging system provides detailed information about:

**Data Processing:**
- Dataset loading progress
- Data preprocessing steps (missing value handling, standardization)
- Data splitting information

**Training:**
- Training request received
- Model creation
- Training start and completion
- Training time and epochs used
- Model evaluation progress

**Errors:**
- Validation errors
- Training failures
- Detailed error messages

All log messages include timestamps and are formatted for easy reading. When troubleshooting issues, check the Docker logs to see exactly where the process is in the pipeline.

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Backend: Kill process on port 8000: `lsof -ti:8000 | xargs kill -9`
   - Frontend: Kill process on port 3000: `lsof -ti:3000 | xargs kill -9`

2. **Docker build failures**
   - Ensure Docker is running: `docker ps`
   - Clear Docker cache: `docker system prune -a`
   - Rebuild without cache: `docker-compose build --no-cache`

3. **Module not found errors**
   - Backend: Ensure package is installed: `pip install -e .`
   - Frontend: Reinstall dependencies: `rm -rf node_modules && npm install`

4. **CORS issues**
   - Ensure backend allows frontend origin in `main.py`
   - Check `NEXT_PUBLIC_API_URL` matches backend URL

5. **Dataset upload fails**
   - Verify file is `.pkl` format
   - Check dataset has exactly 5 features
   - Ensure 'X' and 'y' keys exist in dictionary

6. **Tests fail**
   - Install dev dependencies: `pip install -e ".[dev]"`
   - Ensure virtual environment is activated
   - Check Python version: `python --version` (should be 3.10+)

7. **Training hangs or doesn't complete**
   - Check Docker logs: `docker-compose logs -f backend`
   - Look for log messages indicating where training stopped
   - Verify dataset is valid and not too large for available resources
   - Check for CUDA/GPU errors (should not occur with CPU-only mode)

## Package Structure

```
pydis_nn/
├── data.py          # Data loading, preprocessing, splitting, standardization
├── neuralnetwork.py # NeuralNetwork class for training and prediction
└── utils.py         # Utility functions (e.g., dataset generation)
```

## Development

### Project Structure

```
hb747/
└── interpolator/
    ├── backend/
    │   ├── pydis_nn/        # Core ML package
    │   ├── tests/           # Test suite
    │   ├── main.py          # FastAPI application
    │   ├── Dockerfile       # Backend container
    │   └── pyproject.toml   # Package configuration
    ├── frontend/
    │   ├── src/app/         # Next.js application
    │   ├── Dockerfile       # Frontend container
    │   └── package.json     # Node.js dependencies
    ├── docker-compose.yml   # Service orchestration
    └── README.md           # This file
```

### Code Quality

- **Formatting**: `black` (configured in `pyproject.toml`)
- **Linting**: `flake8` (configured in `pyproject.toml`)
- **Type Checking**: `mypy` (configured in `pyproject.toml`)
- **Testing**: `pytest` with coverage reporting

## Performance and Profiling

This section documents comprehensive performance analysis and benchmarking of the neural network model across different dataset sizes.

### Methodology

Performance benchmarks were conducted using:
- **Dataset sizes**: 1,000 to 10,000 samples (1K increments)
- **Model architecture**: 3 hidden layers [64, 32, 16] neurons
- **Training configuration**: 300 epochs, learning rate 0.001, early stopping disabled
- **Data split**: 70% train, 15% validation, 15% test
- **Random seed**: 42 (for reproducibility)

All benchmarks were run on the same hardware with consistent hyperparameters to ensure comparability. Early stopping was disabled to maintain consistent epoch counts across all tests.

### Training Time Analysis

Training time was measured for datasets ranging from 1K to 10K samples. The model demonstrates efficient sub-linear scaling behavior:

| Dataset Size | Training Time (s) | Time per Sample (ms) |
|--------------|-------------------|----------------------|
| 1,000        | 9.93              | 9.93                 |
| 2,000        | 12.21             | 6.11                 |
| 3,000        | 15.10             | 5.03                 |
| 4,000        | 17.15             | 4.29                 |
| 5,000        | 22.01             | 4.40                 |
| 6,000        | 23.00             | 3.83                 |
| 7,000        | 25.39             | 3.63                 |
| 8,000        | 26.15             | 3.27                 |
| 9,000        | 28.80             | 3.20                 |
| 10,000       | 31.67             | 3.17                 |

**Scaling Behavior:**
- Training time scales sub-linearly with dataset size
- Time per sample decreases as dataset size increases (from 9.93ms to 3.17ms)
- This indicates efficient batch processing and CPU utilization (TensorFlow is configured for CPU-only mode)
- A 10x increase in dataset size (1K → 10K) results in only ~3.2x increase in training time

### Memory Usage

Memory profiling was conducted on a 5,000-sample dataset. Peak memory usage occurs during the training phase:

| Phase | Memory Usage (MB) |
|-------|-------------------|
| After dataset load | 0.35 |
| After preprocessing | 0.54 |
| After model creation | 0.85 |
| **Peak during training** | **4.94** |
| Peak during prediction | 4.94 |

**Key Findings:**
- Peak memory usage is approximately 4.94 MB during training
- Memory usage increases by ~4.09 MB during training phase
- Prediction phase has negligible memory overhead
- Memory usage is efficient and scales well with dataset size

### Accuracy Metrics

Model performance improves significantly with larger datasets, demonstrating the value of more training data:

| Dataset Size | Test R² Score | Test MSE |
|--------------|---------------|----------|
| 1,000        | 0.9417        | 0.017613 |
| 2,000        | 0.9698        | 0.006889 |
| 3,000        | 0.9867        | 0.003629 |
| 4,000        | 0.9929        | 0.001946 |
| 5,000        | 0.9937        | 0.001807 |
| 6,000        | 0.9940        | 0.001784 |
| 7,000        | 0.9965        | 0.001004 |
| 8,000        | 0.9972        | 0.000853 |
| 9,000        | 0.9983        | 0.000482 |
| 10,000       | 0.9985        | 0.000450 |

**Performance Trends:**
- R² score improves from 0.9417 (1K samples) to 0.9985 (10K samples)
- MSE decreases from 0.017613 to 0.000450 (97% reduction)
- Diminishing returns observed after ~7K samples (improvement rate slows)
- Model demonstrates strong generalization with larger datasets

### Conclusions

1. **Scalability**: The model exhibits efficient sub-linear scaling, making it suitable for larger datasets without prohibitive training times.

2. **Memory Efficiency**: Peak memory usage is low (~5 MB), allowing training on systems with limited RAM.

3. **Accuracy Gains**: Significant performance improvements with larger datasets, though returns diminish after ~7K samples.

4. **Training Time**: All dataset sizes (up to 10K) train in under 32 seconds with 300 epochs, meeting the <1 minute requirement for 10K samples.

5. **No Bottlenecks Identified**: Memory and computation scale efficiently without significant bottlenecks.

For detailed results and visualizations, see the benchmark results in `backend/outputs/benchmark_results.json` and `backend/outputs/memory_profile.json`, or run the visualization script:

```bash
cd backend
python scripts/visualize_results.py
```

This generates plots saved to `backend/outputs/` including:
- Training time vs dataset size
- Accuracy metrics vs dataset size
- Scaling analysis (time per sample)
- Memory usage by phase

For comprehensive performance profiling documentation including system specifications, timing methodology, detailed benchmark configuration, and visualization graphs, see the **Performance Profiling** section in the Read the Docs documentation:

**📚 [Performance Profiling Documentation](https://pydis-nn.readthedocs.io/en/latest/performance.html)**

## Version Information

Current version: **1.0.1**

This project uses semantic versioning. The version number is defined in:
- `backend/pyproject.toml`
- `backend/docs/source/conf.py`
- `backend/main.py` (FastAPI app version)

## Git Tags

This project uses git tags to mark major milestones. Tags are created for each completed task:

- `task1-repository-structure` - Task 1: Repository structure
- `task2-package-config` - Task 2: Package configuration
- `task3-data-handling` - Task 3: Data handling module
- `task4-neural-network` - Task 4: Neural network implementation
- `task5-fastapi-backend` - Task 5: FastAPI backend
- `task6-frontend` - Task 6: Next.js frontend
- `task7-testing` - Task 7: Testing and reproducibility
- `task8-performance` - Task 8: Performance profiling
- `task9-documentation` - Task 9: Documentation
- `final` - Final submission version

To view all tags:
```bash
git tag -l
```

To view a specific tag:
```bash
git show <tag-name>
```

## License

Distributed under the MIT License.

## Authors

- Harvey Bermingham

## Use of Generative Tools

This project has utlised auto-generative tools in the development of the frontend and backend of this application.

Example prompts used for this project:
- Generate python code for this plot
- Create a general README.md template structure for this project
- Generate doc-strings for this function
- Generate code to create a button in the Next.js frontend
- Adjust the frontend input boxes to become non-interactive when the training button is pressed
