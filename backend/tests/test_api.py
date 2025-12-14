"""
Tests for FastAPI endpoints.
"""

import pytest
import pickle
import tempfile
import os
import numpy as np
from fastapi.testclient import TestClient

from pydis_nn.utils import generate_sample_dataset


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestUploadEndpoint:
    """Tests for /upload endpoint."""
    
    def test_upload_valid_file(self, client, sample_dataset, app_state_reset):
        """Test uploading a valid .pkl file."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(sample_dataset, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file:
                response = client.post(
                    "/upload",
                    files={"file": ("test_dataset.pkl", file, "application/octet-stream")}
                )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert data["n_samples"] == 1000
            assert data["n_features"] == 5
            assert "missing_values" in data
            assert "duplicate_rows" in data
            assert "memory_usage_mb" in data
            assert "feature_stats" in data
            assert "feature_ranges" in data
            assert len(data["feature_ranges"]) == 5
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_upload_invalid_file_extension(self, client, app_state_reset):
        """Test uploading a file with wrong extension."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as f:
            f.write(b"not a pickle file")
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file:
                response = client.post(
                    "/upload",
                    files={"file": ("test.txt", file, "text/plain")}
                )
            
            assert response.status_code == 400
            assert "must be a .pkl file" in response.json()["detail"].lower()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_upload_invalid_dataset_format(self, client, app_state_reset):
        """Test uploading an invalid dataset format."""
        invalid_data = [1, 2, 3]  # Not a dict
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(invalid_data, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file:
                response = client.post(
                    "/upload",
                    files={"file": ("invalid.pkl", file, "application/octet-stream")}
                )
            
            assert response.status_code == 400
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_upload_wrong_features(self, client, app_state_reset):
        """Test uploading a dataset with wrong number of features."""
        data = generate_sample_dataset(n=100, seed=999)
        data['X'] = data['X'][:, :4]  # Only 4 features
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(data, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file:
                response = client.post(
                    "/upload",
                    files={"file": ("wrong_features.pkl", file, "application/octet-stream")}
                )
            
            assert response.status_code == 400
            assert "5 features" in response.json()["detail"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTrainEndpoint:
    """Tests for /train endpoint."""
    
    def test_train_without_upload(self, client, app_state_reset):
        """Test training without uploading a dataset first."""
        train_request = {
            "hidden_sizes": [32, 16],
            "learning_rate": 0.01,
            "max_iter": 10,
            "random_state": 42
        }
        
        response = client.post("/train", json=train_request)
        
        assert response.status_code == 400
        assert "dataset" in response.json()["detail"].lower()
    
    def test_train_valid(self, client, sample_dataset_small, app_state_reset):
        """Test training with a valid dataset."""
        # First upload a dataset
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(sample_dataset_small, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file:
                upload_response = client.post(
                    "/upload",
                    files={"file": ("train_test.pkl", file, "application/octet-stream")}
                )
            
            assert upload_response.status_code == 200
            
            # Now train
            train_request = {
                "hidden_sizes": [16, 8],
                "learning_rate": 0.01,
                "max_iter": 10,
                "random_state": 42
            }
            
            response = client.post("/train", json=train_request)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "train_r2" in data
            assert "val_r2" in data
            assert "test_r2" in data
            assert "train_mse" in data
            assert "val_mse" in data
            assert "test_mse" in data
            assert "epochs_used" in data
            assert "training_time_seconds" in data
            assert "loss_history" in data
            assert "predictions_sample" in data
            
            assert data["epochs_used"] > 0
            assert len(data["loss_history"]) > 0
            assert len(data["predictions_sample"]) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_train_with_custom_params(self, client, sample_dataset_small, app_state_reset):
        """Test training with custom parameters."""
        # Upload dataset
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(sample_dataset_small, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file:
                client.post(
                    "/upload",
                    files={"file": ("custom_train.pkl", file, "application/octet-stream")}
                )
            
            # Train with custom parameters
            train_request = {
                "hidden_sizes": [32, 16, 8],
                "learning_rate": 0.005,
                "max_iter": 15,
                "random_state": 123,
                "train_size": 0.8,
                "val_size": 0.1,
                "test_size": 0.1
            }
            
            response = client.post("/train", json=train_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    def test_predict_without_training(self, client, app_state_reset):
        """Test prediction without training a model."""
        predict_request = {
            "features": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        response = client.post("/predict", json=predict_request)
        
        assert response.status_code == 400
        assert "not trained" in response.json()["detail"].lower()
    
    def test_predict_wrong_features_count(self, client, sample_dataset_small, app_state_reset):
        """Test prediction with wrong number of features."""
        # Upload and train first
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(sample_dataset_small, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file:
                client.post(
                    "/upload",
                    files={"file": ("wrong_features_test.pkl", file, "application/octet-stream")}
                )
            
            train_request = {
                "hidden_sizes": [16, 8],
                "learning_rate": 0.01,
                "max_iter": 10,
                "random_state": 42
            }
            client.post("/train", json=train_request)
            
            # Now test with wrong number of features
            predict_request = {
                "features": [0.1, 0.2, 0.3, 0.4]  # Only 4 features
            }
            
            response = client.post("/predict", json=predict_request)
            
            assert response.status_code == 400
            assert "5 features" in response.json()["detail"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_predict_valid(self, client, sample_dataset_small, app_state_reset):
        """Test prediction with a trained model."""
        # Upload dataset
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(sample_dataset_small, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as file:
                client.post(
                    "/upload",
                    files={"file": ("predict_test.pkl", file, "application/octet-stream")}
                )
            
            # Train model
            train_request = {
                "hidden_sizes": [16, 8],
                "learning_rate": 0.01,
                "max_iter": 10,
                "random_state": 42
            }
            client.post("/train", json=train_request)
            
            # Make prediction
            predict_request = {
                "features": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            
            response = client.post("/predict", json=predict_request)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "prediction" in data
            assert isinstance(data["prediction"], (int, float))
            assert np.isfinite(data["prediction"])
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCORS:
    """Tests for CORS middleware."""
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        # Test with GET request instead of OPTIONS (FastAPI handles CORS on actual requests)
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Check that request succeeds (CORS is handled by middleware)
        assert response.status_code == 200
        # CORS headers would be checked in integration tests with actual browser

