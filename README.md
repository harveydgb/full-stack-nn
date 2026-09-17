# Neural Network Interpolator

A full-stack system for training and serving neural networks that interpolate
5-dimensional numerical datasets — FastAPI backend, Next.js frontend, Docker Compose
deployment, and a published Python package.

**Research Computing · MPhil in Data Intensive Science, University of Cambridge (2025/26)**

[![PyPI](https://img.shields.io/badge/PyPI-pydis--nn-3775A9)](https://pypi.org/project/pydis-nn/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-8CA1AF)](https://pydis-nn.readthedocs.io/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Next.js](https://img.shields.io/badge/Next.js-20%2B-black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ed)
![Tests](https://img.shields.io/badge/tests-38-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Upload a 5-dimensional dataset through a web interface, train a neural network on it, and
query the trained model for predictions — end to end, with no code.

The project is three things at once: a **published package** (`pydis-nn` on PyPI) containing
the machine learning core; a **REST API** wrapping it; and a **web application** that makes
the whole pipeline usable by someone who has never opened a terminal.

Preprocessing is handled automatically — missing values, standardisation, and train /
validation / test splitting — and TensorFlow is pinned to CPU-only execution so results stay
consistent across hardware.

| | |
| --- | --- |
| **Package** | [`pydis-nn` on PyPI](https://pypi.org/project/pydis-nn/) — v1.0.1 |
| **Documentation** | [pydis-nn.readthedocs.io](https://pydis-nn.readthedocs.io/) — Sphinx, with API reference and guides |
| **Backend** | FastAPI, multi-stage Docker build, health checks, structured logging |
| **Frontend** | Next.js 20 + TypeScript, standalone production build |
| **Tests** | 38 — 16 data handling, 11 neural network, 11 API |

## Quick start

```bash
cd interpolator
./scripts/launch-stack.sh
```

Then open <http://localhost:3000>. The API serves at <http://localhost:8000>, with
interactive Swagger docs at `/docs`.

Requires Docker and Docker Compose. For manual setup, local development, environment
variables and troubleshooting, see **[interpolator/README.md](interpolator/README.md)**.

## Measured performance

Benchmarked from 1,000 to 10,000 samples — three hidden layers [64, 32, 16], 300 epochs,
seed 42, early stopping disabled so epoch counts stay comparable.

**Training scales sub-linearly.** A 10× increase in data costs only ~3.2× the time, because
per-sample cost falls as batches fill:

| Samples | Training time | Per sample | Test R² | Test MSE |
| --- | --- | --- | --- | --- |
| 1,000 | 9.93 s | 9.93 ms | 0.9417 | 0.017613 |
| 5,000 | 22.01 s | 4.40 ms | 0.9937 | 0.001807 |
| 10,000 | 31.67 s | 3.17 ms | **0.9985** | **0.000450** |

Accuracy improves throughout — R² from 0.942 to 0.999, MSE down 97% — though returns
visibly diminish beyond about 7,000 samples.

**Peak memory is 4.94 MB** on a 5,000-sample run, almost all of it during training;
prediction adds essentially nothing. The whole thing trains 10,000 samples in under
32 seconds on CPU.

Full methodology, system specifications and plots:
[performance documentation](https://pydis-nn.readthedocs.io/en/latest/performance.html).

## Repository structure

```
.
├── .readthedocs.yaml
└── interpolator/
    ├── backend/
    │   ├── pydis_nn/        # Core package — data, neuralnetwork, utils, logger
    │   ├── tests/           # 38 tests
    │   ├── docs/            # Sphinx source
    │   ├── scripts/         # Benchmarking, memory profiling, visualisation
    │   ├── main.py          # FastAPI application
    │   └── Dockerfile       # Multi-stage: development and production targets
    ├── frontend/
    │   ├── src/app/         # Next.js application
    │   └── Dockerfile       # Multi-stage: deps, builder, runner
    ├── scripts/             # launch-stack, docker-start/stop/logs, test-pipeline
    ├── docker-compose.yml
    └── README.md            # Full documentation
```

## Dataset format

Pickled dictionaries with exactly five features:

```python
{
    'X': numpy.ndarray,  # (n_samples, 5)
    'y': numpy.ndarray   # (n_samples,)
}
```

NaN and infinite values are handled automatically. `pydis_nn.utils.generate_sample_dataset`
produces a valid example.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check |
| `POST /upload` | Upload a `.pkl` dataset |
| `POST /train` | Train a model |
| `POST /predict` | Predict with the trained model |

## Tests

```bash
cd interpolator/backend
pip install -e ".[dev]"
pytest tests/ -v --cov=pydis_nn
```

An end-to-end check of the full stack — bringing both services up, waiting for health, and
exercising the API — runs via `interpolator/scripts/test-pipeline.sh`.

## Use of Generative Tools

This project has utlised auto-generative tools in the development of the frontend and backend
of this application.

Example prompts used for this project:

- Generate python code for this plot
- Create a general README.md template structure for this project
- Generate doc-strings for this function
- Generate code to create a button in the Next.js frontend
- Adjust the frontend input boxes to become non-interactive when the training button is pressed

## Author

Harvey Bermingham — MPhil in Data Intensive Science, University of Cambridge

## License

Released under the MIT License. See [LICENSE](LICENSE).
