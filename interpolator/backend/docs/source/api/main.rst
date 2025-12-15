FastAPI Application
===================

The ``main`` module contains the FastAPI application with REST API endpoints.

Endpoints
---------

.. http:get:: /health

   Health check endpoint.

   **Response:**
   
   .. sourcecode:: json
   
      {
        "status": "healthy"
      }

.. http:post:: /upload

   Upload a .pkl dataset file.
   
   **Request:** Multipart form data with ``file`` field containing .pkl file
   
   **Response:**
   
   .. sourcecode:: json
   
      {
        "status": "success",
        "message": "Dataset uploaded successfully",
        "n_samples": 5000,
        "n_features": 5,
        "missing_values": 0,
        "duplicate_rows": 0,
        "memory_usage_mb": 0.15,
        "feature_stats": {...},
        "feature_ranges": [...]
      }

.. http:post:: /train

   Train a neural network model on the uploaded dataset.
   
   **Request Body:**
   
   .. sourcecode:: json
   
      {
        "hidden_sizes": [64, 32, 16],
        "learning_rate": 0.001,
        "max_iter": 300,
        "random_state": 42,
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "standardize": true
      }
   
   **Response:**
   
   .. sourcecode:: json
   
      {
        "status": "success",
        "train_r2": 0.99,
        "val_r2": 0.98,
        "test_r2": 0.97,
        "epochs_used": 300,
        "training_time_seconds": 22.5,
        "loss_history": [...],
        "predictions_sample": [...]
      }

.. http:post:: /predict

   Make a prediction using the trained model.
   
   **Request Body:**
   
   .. sourcecode:: json
   
      {
        "features": [0.1, 0.2, 0.3, 0.4, 0.5]
      }
   
   **Response:**
   
   .. sourcecode:: json
   
      {
        "status": "success",
        "prediction": 0.12345
      }

Module Reference
----------------

.. automodule:: main
   :members:
   :undoc-members:
   :show-inheritance:

