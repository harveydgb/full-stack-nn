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

from pydis_nn.data import load_dataset, load_and_preprocess, load_raw_dataset, calculate_dataset_statistics
from pydis_nn.neuralnetwork import NeuralNetwork
from pydis_nn.logger import Logger

# Initialize logger
logger = Logger()

app = FastAPI(
    title="5D Neural Network Interpolator API",
    description="API for training and querying 5D neural network interpolation models",
    version="1.0.1",
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
class FeatureStats(BaseModel):
    min_avg: float
    max_avg: float
    target_mean: float
    target_std: float
    target_min: float
    target_max: float

class FeatureRange(BaseModel):
    min: float
    max: float

class UploadResponse(BaseModel):
    status: str
    message: str
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    missing_values: Optional[int] = None
    duplicate_rows: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    feature_stats: Optional[FeatureStats] = None
    feature_ranges: Optional[List[FeatureRange]] = None


class TrainRequest(BaseModel):
    hidden_sizes: List[int] = [64, 32, 16]
    learning_rate: float = 0.001
    max_iter: int = 300
    random_state: int = 42
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15


class LossHistoryItem(BaseModel):
    epoch: int
    loss: float
    val_loss: float

class PredictionSample(BaseModel):
    true: float
    pred: float

class TrainResponse(BaseModel):
    status: str
    message: str
    train_r2: Optional[float] = None
    val_r2: Optional[float] = None
    test_r2: Optional[float] = None
    train_mse: Optional[float] = None
    val_mse: Optional[float] = None
    test_mse: Optional[float] = None
    epochs_used: Optional[int] = None
    training_time_seconds: Optional[float] = None
    loss_history: Optional[List[LossHistoryItem]] = None
    predictions_sample: Optional[List[PredictionSample]] = None


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
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load raw data for pre-processing statistics
        raw_data = load_raw_dataset(tmp_path)
        X_raw = raw_data['X']
        y_raw = raw_data['y']
        
        # Load processed data for post-processing statistics
        data = load_dataset(tmp_path)
        X = data['X']
        y = data['y']
        
        # Calculate all statistics using data module function
        stats = calculate_dataset_statistics(X_raw, y_raw, X, y)
        
        # Clean up old file if exists (silent failure is OK)
        if app.state.dataset_path and os.path.exists(app.state.dataset_path):
            try:
                os.unlink(app.state.dataset_path)
            except (OSError, PermissionError):
                pass
        
        # Store file path for training
        app.state.dataset_path = tmp_path
        app.state.model = None
        app.state.scaler = None
        tmp_path = None  # Prevent cleanup since file is now managed by app.state
        
        return UploadResponse(
            status="success",
            message="Dataset uploaded successfully",
            n_samples=X.shape[0],
            n_features=X.shape[1],
            missing_values=stats['missing_values'],
            duplicate_rows=stats['duplicate_rows'],
            memory_usage_mb=stats['memory_usage_mb'],
            feature_stats=FeatureStats(**stats['feature_stats']),
            feature_ranges=[FeatureRange(**fr) for fr in stats['feature_ranges']]
        )
    
    except (ValueError, FileNotFoundError) as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train a neural network model on the uploaded dataset.
    
    Requires a dataset to be uploaded first via /upload endpoint.
    """
    logger.info("Training request received")
    
    if app.state.dataset_path is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload a dataset first using /upload endpoint."
        )
    
    if not os.path.exists(app.state.dataset_path):
        raise HTTPException(status_code=400, detail="Dataset file not found")
    
    try:
        logger.info("Starting data preprocessing")
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
        
        logger.info("Creating neural network model")
        model = NeuralNetwork(
            hidden_sizes=request.hidden_sizes,
            learning_rate=request.learning_rate,
            max_iter=request.max_iter,
            random_state=request.random_state
        )
        
        # Train model with history tracking
        logger.info(f"Starting training with max_iter={request.max_iter}")
        start_time = time.time()
        model, loss_history_list = model.fit(
            data['X_train'],
            data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            return_history=True
        )
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Get actual epochs run (may be less than max_iter if early stopping triggered)
        epochs_used = model._epochs_run
        
        logger.info("Evaluating model performance")
        # Evaluate model performance
        metrics = model.evaluate_all(
            data['X_train'], data['y_train'],
            X_val=data['X_val'], y_val=data['y_val'],
            X_test=data['X_test'], y_test=data['y_test']
        )
        
        train_r2 = metrics['train_r2']
        val_r2 = metrics.get('val_r2', 0.0)
        test_r2 = metrics.get('test_r2', 0.0)
        train_mse = metrics['train_mse']
        val_mse = metrics.get('val_mse', 0.0)
        test_mse = metrics.get('test_mse', 0.0)
        
        # Sample predictions for scatter plot (take up to 100 samples from test set)
        test_pred = model.predict(data['X_test'])
        test_sample_size = min(100, len(data['y_test']))
        sample_indices = np.random.choice(len(data['y_test']), test_sample_size, replace=False)
        predictions_sample = [
            PredictionSample(true=float(data['y_test'][i]), pred=float(test_pred[i]))
            for i in sample_indices
        ]
        
        # Store trained model
        app.state.model = model
        logger.info("Training completed successfully")
        
        return TrainResponse(
            status="success",
            message="Model trained successfully",
            train_r2=float(train_r2),
            val_r2=float(val_r2),
            test_r2=float(test_r2),
            train_mse=train_mse,
            val_mse=val_mse,
            test_mse=test_mse,
            epochs_used=epochs_used,
            training_time_seconds=float(training_time),
            loss_history=[LossHistoryItem(**item) for item in loss_history_list],
            predictions_sample=predictions_sample
        )
    
    except ValueError as e:
        logger.error(f"Training validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
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
        
        # Extract scalar value from numpy array
        pred_value = float(prediction[0])
        
        return PredictResponse(
            status="success",
            prediction=pred_value
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
