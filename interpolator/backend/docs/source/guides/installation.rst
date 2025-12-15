Installation Guide
==================

This guide covers installation and setup of the 5D Neural Network Interpolator system.

Prerequisites
-------------

* **Python** 3.10 or higher (for backend)
* **Node.js** 20 or higher (for frontend)
* **Docker & Docker Compose** (optional, recommended for production)

Option 1: Docker Compose (Recommended)
---------------------------------------

The easiest way to get started is using Docker Compose, which sets up both the backend and frontend services automatically.

1. **Start the services:**

   .. code-block:: bash

      cd interpolator
      docker-compose up --build

2. **Access the services:**

   * Frontend UI: http://localhost:3000
   * Backend API: http://localhost:8000
   * API Docs (Swagger): http://localhost:8000/docs
   * API Docs (ReDoc): http://localhost:8000/redoc

3. **Stop the services:**

   .. code-block:: bash

      docker-compose down

Option 2: Manual Setup
----------------------

Backend Setup
~~~~~~~~~~~~~

1. **Create and activate virtual environment:**

   .. code-block:: bash

      cd interpolator/backend
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

2. **Install package in editable mode:**

   .. code-block:: bash

      pip install -e .

   This will install all dependencies including:
   * numpy
   * tensorflow
   * fastapi
   * scikit-learn
   * and others (see ``pyproject.toml``)

3. **Start development server:**

   .. code-block:: bash

      uvicorn main:app --reload --host 0.0.0.0 --port 8000

   The backend will be available at http://localhost:8000

Frontend Setup
~~~~~~~~~~~~~~

1. **Install dependencies:**

   .. code-block:: bash

      cd interpolator/frontend
      npm install

2. **Start development server:**

   .. code-block:: bash

      npm run dev

   The frontend will be available at http://localhost:3000

Environment Variables
---------------------

Backend
~~~~~~~

* ``PYTHONPATH``: Python module search path (default: ``/app`` in Docker)
* ``PYTHONUNBUFFERED``: Disable Python output buffering (default: ``1``)

Frontend
~~~~~~~~

* ``NODE_ENV``: Node.js environment (default: ``production``)
* ``NEXT_PUBLIC_API_URL``: Backend API base URL (default: ``http://localhost:8000``)

Verification
------------

To verify the installation:

1. Check backend health: http://localhost:8000/health
2. Check frontend: http://localhost:3000
3. Check API docs: http://localhost:8000/docs

Troubleshooting
---------------

**Port already in use:**
   * Backend (8000): ``lsof -ti:8000 | xargs kill -9``
   * Frontend (3000): ``lsof -ti:3000 | xargs kill -9``

**Module not found errors:**
   * Ensure virtual environment is activated
   * Run ``pip install -e .`` from the backend directory

**Docker build failures:**
   * Ensure Docker is running: ``docker ps``
   * Clear Docker cache: ``docker system prune -a``
   * Rebuild without cache: ``docker-compose build --no-cache``

