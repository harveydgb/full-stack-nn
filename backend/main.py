"""
FastAPI backend for 5D neural network interpolation system.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import time
import numpy as np

from pydis_nn.data import load_dataset, load_and_preprocess
from pydis_nn.neuralnetwork import NeuralNetwork

app = FastAPI(
    title="5D Neural Network Interpolator API",
    description="API for training and querying 5D neural network interpolation models",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State management for dataset, model, and scaler
app.state.dataset_path = None
app.state.model = None
app.state.scaler = None


# Pydantic models for requests/responses
class UploadResponse(BaseModel):
    status: str
    message: str
    n_samples: Optional[int] = None
    n_features: Optional[int] = None


class TrainRequest(BaseModel):
    hidden_sizes: List[int] = [64, 32, 16]
    learning_rate: float = 0.001
    max_iter: int = 1000
    random_state: int = 42
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    standardize: bool = True


class TrainResponse(BaseModel):
    status: str
    message: str
    train_r2: Optional[float] = None
    val_r2: Optional[float] = None
    test_r2: Optional[float] = None
    training_time_seconds: Optional[float] = None


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    status: str
    prediction: float


@app.get("/")
async def root():
    return {"message": "Neural Network Interpolator API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a .pkl dataset file with 'X' and 'y' keys. X must have 5 features."""
    if not file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="File must be a .pkl file")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            data = load_dataset(tmp_path)
            X = data['X']
            
            # Store file path for training and clean up old file if exists
            if app.state.dataset_path and os.path.exists(app.state.dataset_path):
                os.unlink(app.state.dataset_path)
            
            # Resetting when new dataset uploaded
            app.state.dataset_path = tmp_path
            app.state.model = None
            app.state.scaler = None
            
            return UploadResponse(
                status="success",
                message="Dataset uploaded successfully",
                n_samples=X.shape[0],
                n_features=X.shape[1]
            )
        except (ValueError, FileNotFoundError) as e:
            # Clean up temp file on validation error
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train a neural network model on the uploaded dataset.
    
    Requires a dataset to be uploaded first via /upload endpoint.
    """
    if app.state.dataset_path is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload a dataset first using /upload endpoint."
        )
    
    if not os.path.exists(app.state.dataset_path):
        raise HTTPException(status_code=400, detail="Dataset file not found")
    
    try:
        data = load_and_preprocess(
            app.state.dataset_path,
            train_size=request.train_size,
            val_size=request.val_size,
            test_size=request.test_size,
            standardize=True,
            random_state=request.random_state
        )
        
        # Store scaler for prediction if standardization was used
        app.state.scaler = data.get('scaler')
        
        model = NeuralNetwork(
            hidden_sizes=request.hidden_sizes,
            learning_rate=request.learning_rate,
            max_iter=request.max_iter,
            random_state=request.random_state
        )
        
        start_time = time.time()
        model.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
        training_time = time.time() - start_time
        
        # Calculate R² scores
        train_r2 = model.score(data['X_train'], data['y_train'])
        val_r2 = model.score(data['X_val'], data['y_val'])
        test_r2 = model.score(data['X_test'], data['y_test'])
        
        # Store trained model
        app.state.model = model
        
        return TrainResponse(
            status="success",
            message="Model trained successfully",
            train_r2=float(train_r2),
            val_r2=float(val_r2),
            test_r2=float(test_r2),
            training_time_seconds=float(training_time)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make a prediction using the trained model.
    
    Requires a model to be trained first via /train endpoint.
    """
    if app.state.model is None:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    if len(request.features) != 5:
        raise HTTPException(status_code=400, detail=f"Need 5 features, got {len(request.features)}")
    
    try:
        # Convert to numpy array, reshape to (1, 5) for single prediction
        X = np.array([request.features], dtype=np.float32)
        
        # Apply scaler transformation if standardization was used during training
        if app.state.scaler is not None:
            X = app.state.scaler.transform(X)
        
        # Make prediction
        prediction = app.state.model.predict(X)
        
        # prediction is already a numpy array, get scalar value
        pred_value = float(prediction[0]) if len(prediction.shape) > 0 else float(prediction)
        
        return PredictResponse(
            status="success",
            prediction=pred_value
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
