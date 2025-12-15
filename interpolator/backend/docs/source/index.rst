5D Neural Network Interpolator
===============================

A full-stack research-grade system for learning and serving neural network models that can interpolate any 5-dimensional numerical dataset.

Overview
--------

This project implements a complete machine learning pipeline:

* **Backend**: FastAPI server with neural network training and prediction endpoints
* **Frontend**: Next.js web application with an interactive UI for dataset management and model training
* **Neural Network**: Lightweight TensorFlow/Keras implementation optimized for 5D regression tasks

The system automatically handles data preprocessing (missing values, standardization), train/validation/test splits, and provides comprehensive metrics and visualizations.

Quick Start
-----------

.. code-block:: bash

   # Start the entire stack
   cd interpolator
   ./scripts/launch-stack.sh
   
   # Or manually with Docker Compose
   docker-compose up --build

The services will be available at:

* **Frontend**: http://localhost:3000
* **Backend API**: http://localhost:8000
* **API Documentation**: http://localhost:8000/docs

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/installation
   guides/usage
   guides/dataset

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Testing

   testing/index

.. toctree::
   :maxdepth: 2
   :caption: Performance

   performance

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

