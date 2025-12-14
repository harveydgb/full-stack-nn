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

from pydis_nn.data import load_dataset, load_and_preprocess, load_raw_dataset
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
    max_iter: int = 1000
    random_state: int = 42
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    standardize: bool = True


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
    train_rmse: Optional[float] = None
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
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Load raw data to compute pre-standardization ranges
            raw_data = load_raw_dataset(tmp_path)
            X_raw = raw_data['X']
            
            # Compute feature ranges from raw data (pre-standardization, pre-missing value handling)
            feature_ranges = [
                FeatureRange(
                    min=float(np.nanmin(X_raw[:, i])), 
                    max=float(np.nanmax(X_raw[:, i]))
                )
                for i in range(X_raw.shape[1])
            ]
            
            # Now load cleaned data for other statistics
            data = load_dataset(tmp_path)
            X = data['X']
            y = data['y']
            
            # Compute dataset statistics (X and y are cleaned by load_dataset, missing values handled)
            missing_values = int(np.isnan(X).sum() + np.isnan(y).sum())
            duplicate_rows = int(len(X) - len(np.unique(X, axis=0)))
            memory_usage_mb = float((X.nbytes + y.nbytes) / (1024 * 1024))
            
            # Compute feature stats
            feature_stats = FeatureStats(
                min_avg=float(X.min(axis=0).mean()),
                max_avg=float(X.max(axis=0).mean()),
                target_mean=float(y.mean()),
                target_std=float(y.std()),
                target_min=float(y.min()),
                target_max=float(y.max())
            )
            
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
                n_features=X.shape[1],
                missing_values=missing_values,
                duplicate_rows=duplicate_rows,
                memory_usage_mb=memory_usage_mb,
                feature_stats=feature_stats,
                feature_ranges=feature_ranges
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
        
        # Custom callback to capture loss history
        from tensorflow.keras.callbacks import Callback
        loss_history_list = []
        
        class LossHistory(Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is not None:
                    loss_history_list.append({
                        'epoch': epoch,
                        'loss': float(logs.get('loss', 0)),
                        'val_loss': float(logs.get('val_loss', logs.get('loss', 0)))
                    })
        
        # Prepare data and call model.fit directly with callback
        X_train_float = data['X_train'].astype(np.float32)
        y_train_float = data['y_train'].astype(np.float32)
        X_val_float = data['X_val'].astype(np.float32)
        y_val_float = data['y_val'].astype(np.float32)
        
        start_time = time.time()
        history_callback = LossHistory()
        model.model.fit(
            X_train_float, y_train_float,
            epochs=request.max_iter,
            batch_size=32,
            verbose=0,
            validation_data=(X_val_float, y_val_float),
            callbacks=[history_callback]
        )
        training_time = time.time() - start_time
        
        # Calculate R² scores
        train_r2 = model.score(data['X_train'], data['y_train'])
        val_r2 = model.score(data['X_val'], data['y_val'])
        test_r2 = model.score(data['X_test'], data['y_test'])
        
        # Calculate MSE and RMSE
        from sklearn.metrics import mean_squared_error
        train_pred = model.predict(data['X_train'])
        val_pred = model.predict(data['X_val'])
        test_pred = model.predict(data['X_test'])
        
        train_mse = float(mean_squared_error(data['y_train'], train_pred))
        val_mse = float(mean_squared_error(data['y_val'], val_pred))
        test_mse = float(mean_squared_error(data['y_test'], test_pred))
        train_rmse = float(np.sqrt(train_mse))
        
        # Sample predictions for scatter plot (take up to 100 samples from test set)
        test_sample_size = min(100, len(data['y_test']))
        sample_indices = np.random.choice(len(data['y_test']), test_sample_size, replace=False)
        predictions_sample = [
            PredictionSample(true=float(data['y_test'][i]), pred=float(test_pred[i]))
            for i in sample_indices
        ]
        
        # Store trained model
        app.state.model = model
        
        return TrainResponse(
            status="success",
            message="Model trained successfully",
            train_r2=float(train_r2),
            val_r2=float(val_r2),
            test_r2=float(test_r2),
            train_mse=train_mse,
            val_mse=val_mse,
            test_mse=test_mse,
            train_rmse=train_rmse,
            training_time_seconds=float(training_time),
            loss_history=[LossHistoryItem(**item) for item in loss_history_list],
            predictions_sample=predictions_sample
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
