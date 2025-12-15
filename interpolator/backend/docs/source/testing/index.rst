Test Suite
===========

This section describes the test suite for the 5D Neural Network Interpolator.

Test Structure
--------------

The test suite is located in ``backend/tests/`` and uses `pytest <https://pytest.org/>`_ as the testing framework.

Directory Structure
-------------------

::

   backend/tests/
   ├── __init__.py
   ├── conftest.py          # Pytest fixtures and configuration
   ├── test_data.py         # Tests for pydis_nn.data module
   ├── test_neuralnetwork.py # Tests for pydis_nn.neuralnetwork module
   ├── test_api.py          # Tests for FastAPI endpoints
   └── fixtures/            # Test data files (if needed)

Running Tests
-------------

Install dev dependencies:

.. code-block:: bash

   cd interpolator/backend
   pip install -e ".[dev]"

Run all tests:

.. code-block:: bash

   pytest tests/ -v

Run specific test file:

.. code-block:: bash

   pytest tests/test_data.py -v

Run with coverage:

.. code-block:: bash

   pytest tests/ -v --cov=pydis_nn --cov-report=html

View coverage report:

.. code-block:: bash

   open htmlcov/index.html  # macOS
   xdg-open htmlcov/index.html  # Linux

Test Coverage
-------------

The test suite includes:

**Data Module Tests (17 tests)**
   * Loading raw datasets
   * Dataset validation (5D requirement)
   * Missing value handling
   * Data splitting (train/val/test)
   * Feature standardization
   * Integration pipeline tests

**Neural Network Tests (13 tests)**
   * Model initialization
   * Training with and without validation data
   * Prediction functionality
   * R² score calculation
   * Early stopping behavior
   * Model state management

**API Endpoint Tests (12 tests)**
   * Health check endpoint
   * Dataset upload (valid/invalid files)
   * Model training endpoint
   * Prediction endpoint
   * Error handling
   * CORS configuration
   * State management

Test Fixtures
-------------

Common fixtures are defined in ``conftest.py``:

* ``sample_dataset``: Generated 5D dataset (1000 samples)
* ``sample_dataset_small``: Small dataset for faster tests (100 samples)
* ``temp_pkl_file``: Temporary .pkl file with sample data
* ``trained_model``: Pre-trained NeuralNetwork instance
* ``client``: FastAPI TestClient instance
* ``app_state_reset``: Fixture to reset application state

Example Test
------------

.. code-block:: python

   def test_load_dataset_valid(temp_pkl_file):
       """Test loading a valid 5D dataset."""
       from pydis_nn.data import load_dataset
       
       data = load_dataset(temp_pkl_file)
       assert data['X'].shape[1] == 5
       assert data['X'].shape[0] == data['y'].shape[0]

Testing Strategy
----------------

* **Unit Tests**: Test individual functions and classes in isolation
* **Integration Tests**: Test API endpoints with full request/response cycle
* **Fixtures**: Reusable test data and setup/teardown logic
* **Coverage**: Aim for >80% code coverage on critical paths

Continuous Integration
----------------------

Tests are designed to run in CI/CD pipelines:

* No external dependencies required
* All tests use temporary files or in-memory data
* Tests are deterministic (fixed random seeds)
* Fast execution (< 30 seconds for full suite)

