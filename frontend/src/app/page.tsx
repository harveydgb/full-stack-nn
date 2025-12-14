"use client";

import { useState, useEffect, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  ReferenceLine,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Updated to include characterization metrics
type UploadResponse = {
  status: string;
  message: string;
  n_samples?: number;
  n_features?: number;
  // New Dataset Characterization Metrics
  missing_values?: number;
  duplicate_rows?: number;
  memory_usage_mb?: number;
  feature_stats?: {
    min_avg: number;
    max_avg: number;
    target_mean: number;
    target_std: number;
    target_min: number;
    target_max: number;
  };
  feature_ranges?: Array<{
    min: number;
    max: number;
  }>;
};

type TrainRequest = {
  hidden_sizes: number[];
  learning_rate: number;
  max_iter: number;
  random_state: number;
  train_size: number;
  val_size: number;
  test_size: number;
  standardize: boolean;
};

type TrainResponse = {
  status: string;
  message: string;
  // Metrics
  train_r2?: number;
  val_r2?: number;
  test_r2?: number;
  train_mse?: number;
  val_mse?: number;
  test_mse?: number;
  train_rmse?: number;
  training_time_seconds?: number;
  
  // Graph Data
  loss_history?: Array<{ epoch: number; loss: number; val_loss: number }>;
  predictions_sample?: Array<{ true: number; pred: number }>;
};

type PredictResponse = {
  status: string;
  prediction: number;
};

export default function Page() {
  const [currentStep, setCurrentStep] = useState(1);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
  // Updated state type to hold full response
  const [datasetInfo, setDatasetInfo] = useState<UploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  // New state for Upload accordion
  const [uploadResultsExpanded, setUploadResultsExpanded] = useState(false);

  const [trainingParams, setTrainingParams] = useState<TrainRequest>({
    hidden_sizes: [64, 32, 16],
    learning_rate: 0.001,
    max_iter: 1000,
    random_state: 42,
    train_size: 0.7,
    val_size: 0.15,
    test_size: 0.15,
    standardize: true,
  });
  const [trainingStatus, setTrainingStatus] = useState<"idle" | "training" | "success" | "error">("idle");
  const [trainingResults, setTrainingResults] = useState<TrainResponse | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [resultsExpanded, setResultsExpanded] = useState(false);

  const [features, setFeatures] = useState<[number, number, number, number, number]>([0, 0, 0, 0, 0]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predictStatus, setPredictStatus] = useState<"idle" | "predicting" | "success" | "error">("idle");
  const [predictError, setPredictError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (trainingStatus === "success") {
      setResultsExpanded(true);
    }
  }, [trainingStatus]);

  useEffect(() => {
    // Auto-expand upload results when upload finishes
    if (uploadStatus === "success") {
      setUploadResultsExpanded(true);
    }
  }, [uploadStatus]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setUploadStatus("idle");
      setUploadError(null);
      setDatasetInfo(null);
      setUploadResultsExpanded(false);
    }
  };

  const handleUpload = async () => {
    if (!uploadedFile) return;

    setUploadStatus("uploading");
    setUploadError(null);

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = (await res.json()) as UploadResponse;

      if (!res.ok || data.status !== "success") {
        throw new Error(data.message || "Upload failed");
      }

      setUploadStatus("success");
      setDatasetInfo(data);
      
    } catch (e: any) {
      setUploadStatus("error");
      setUploadError(e.message || "Upload failed");
    }
  };

  const generateMockGraphData = () => {
    const loss_history = Array.from({ length: 50 }, (_, i) => ({
      epoch: i * 20,
      loss: 0.5 * Math.exp(-0.1 * i) + 0.05 + Math.random() * 0.02,
      val_loss: 0.6 * Math.exp(-0.1 * i) + 0.08 + Math.random() * 0.03,
    }));

    const predictions_sample = Array.from({ length: 50 }, () => {
      const trueVal = Math.random() * 10;
      return {
        true: trueVal,
        pred: trueVal + (Math.random() - 0.5) * 2,
      };
    });

    return { loss_history, predictions_sample };
  };

  const handleTrain = async () => {
    // Validate parameters before training
    const validationError = validateTrainingParams(trainingParams);
    if (validationError) {
      setTrainingStatus("error");
      setTrainingError(validationError);
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setTrainingStatus("training");
    setTrainingError(null);
    setResultsExpanded(false);

    try {
      const res = await fetch(`${API_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...trainingParams, standardize: true }),
        signal,
      });

      const data = (await res.json()) as TrainResponse;

        if (!res.ok || data.status !== "success") {
          throw new Error(data.message || "Training failed");
        }

        setTrainingStatus("success");
        setTrainingResults(data);
      abortControllerRef.current = null;
    } catch (e: any) {
      if (e.name === 'AbortError') {
        setTrainingStatus("idle");
        setTrainingError(null);
      } else {
        setTrainingStatus("error");
        setTrainingError(e.message || "Training failed");
      }
      abortControllerRef.current = null;
    }
  };

  const handlePredict = async () => {
    setPredictStatus("predicting");
    setPredictError(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });

      const data = (await res.json()) as PredictResponse;

      if (!res.ok || data.status !== "success") {
        throw new Error("Prediction failed");
      }

      setPredictStatus("success");
      setPrediction(data.prediction);
    } catch (e: any) {
      setPredictStatus("error");
      setPredictError(e.message || "Prediction failed");
    }
  };

  const validateTrainingParams = (params: TrainRequest): string | null => {
    // Validate hidden layers: must be positive, max 6 layers
    if (params.hidden_sizes.length > 6) {
      return "Maximum 6 hidden layers allowed";
    }
    for (const size of params.hidden_sizes) {
      if (size <= 0) {
        return "Hidden layer sizes must be positive";
      }
      if (size > 1024) {
        return "Maximum 1024 neurons per layer";
      }
    }
    
    // Validate learning rate: must be positive and <= 1
    if (params.learning_rate <= 0) {
      return "Learning rate must be positive";
    }
    if (params.learning_rate > 1) {
      return "Learning rate must be <= 1";
    }
    
    // Validate max_iter: must be between 5 and 1000
    if (params.max_iter < 5 || params.max_iter > 1000) {
      return "Max iterations must be between 5 and 1000";
    }
    
    // Validate train/val/test sizes are positive
    if (params.train_size <= 0 || params.val_size <= 0 || params.test_size <= 0) {
      return "Train, validation, and test sizes must be positive";
    }
    
    return null;
  };

  const updateFeature = (index: number, value: string) => {
    const newFeatures = [...features] as [number, number, number, number, number];
    newFeatures[index] = parseFloat(value) || 0;
    setFeatures(newFeatures);
  };

  const handleReset = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setCurrentStep(1);
    setUploadedFile(null);
    setUploadStatus("idle");
    setDatasetInfo(null);
    setUploadError(null);
    setUploadResultsExpanded(false);
    setTrainingStatus("idle");
    setTrainingResults(null);
    setTrainingError(null);
    setPrediction(null);
    setPredictStatus("idle");
    setPredictError(null);
    setResultsExpanded(false);
  };

  const steps = ["Upload", "Train", "Predict"];

  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      <div className="fixed inset-0 opacity-40 pointer-events-none">
        <div className="mesh-gradient"></div>
      </div>

      <div className="fixed top-0 left-0 right-0 bg-white/70 backdrop-blur-2xl border-b border-gray-200/50 z-50">
        <div className="max-w-2xl mx-auto px-8 py-6 relative">
          <div className="text-center mb-4">
            <h1 className="text-3xl font-semibold text-black tracking-tight">
              5D Neural Network Interpolator
            </h1>
          </div>
          <div className="relative">
            <div className="flex items-center justify-center">
              <div className="flex items-center flex-1">
                {steps.map((step, idx) => (
                  <div key={idx} className="flex items-center" style={{ flex: idx === steps.length - 1 ? '0 0 auto' : '1 1 0%' }}>
                    <div className="flex flex-col items-center">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold transition-all duration-300 ${
                          idx + 1 < currentStep
                            ? "bg-black text-white scale-100"
                            : idx + 1 === currentStep
                            ? "bg-black text-white scale-110"
                            : "bg-gray-200/80 text-gray-400 scale-100"
                        }`}
                      >
                        {idx + 1 < currentStep ? (
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        ) : (
                          idx + 1
                        )}
                      </div>
                      <span
                        className={`text-xs mt-2 font-medium transition-all duration-300 whitespace-nowrap ${
                          idx + 1 <= currentStep ? "text-black" : "text-gray-400"
                        }`}
                      >
                        {step}
                      </span>
                    </div>
                    {idx < steps.length - 1 && (
                      <div className="h-0.5 mx-6 flex-1 relative" style={{ alignSelf: 'flex-start', marginTop: '16px' }}>
                        <div className="h-full bg-gray-200/80 absolute inset-0" />
                        <div
                          className={`h-full transition-all duration-700 ease-out relative ${
                            idx + 1 < currentStep ? "bg-black w-full" : "bg-transparent w-0"
                          }`}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleReset}
          className="absolute right-10 bottom-21 px-4 py-2.5 text-sm font-semibold text-gray-700 hover:text-black transition-colors bg-white/70 backdrop-blur-xl border border-gray-200/50 rounded-lg hover:bg-white/90"
        >
          Reset
        </button>
      </div>

      <div className="pt-25 pb-20 px-8 relative z-10">
        <div className="max-w-2xl mx-auto">
          {/* Step 1: Upload */}
          {currentStep === 1 && (
            <div className="animate-fadeIn space-y-8">
              <div className="text-center pt-20">
                <h2 className="text-2xl font-semibold text-black mb-2 tracking-tight">Upload your data</h2>
                <p className="text-sm text-gray-500 font-light">
                  Start by uploading a .pkl file containing your dataset
                </p>
              </div>

              <div className="space-y-6">
                <label className="block cursor-pointer group relative">
                  <div className={`glass-panel border-2 rounded-3xl p-16 text-center transition-all duration-500 ease-in-out overflow-hidden
                    ${uploadStatus === 'error' ? 'border-red-200/50 bg-red-50/10' : 
                      uploadStatus === 'success' ? 'border-green-200/50 bg-green-50/10' : 
                      uploadedFile ? 'border-black/20 bg-gray-50/30' : 'border-gray-200/50 hover:border-gray-300/80 hover:bg-gray-50/30'}`}>
                    
                    <input
                      type="file"
                      accept=".pkl"
                      onChange={handleFileChange}
                      disabled={uploadStatus === 'success' || uploadStatus === 'uploading'}
                      className="hidden"
                    />

                    <div className="mb-6 relative h-20 w-20 mx-auto flex items-center justify-center">
                      {uploadStatus === "idle" && (
                        <svg className={`w-20 h-20 transition-all duration-300 ${uploadedFile ? 'text-black scale-105' : 'text-gray-300 group-hover:text-gray-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                      )}
                      {uploadStatus === "uploading" && (
                        <svg className="w-20 h-20 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                          <path className="cloud-pulse" strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3" />
                          <path className="arrow-rise" strokeLinecap="round" strokeLinejoin="round" d="M12 17v-9m0 0l-3 3m3-3l3 3" />
                        </svg>
                      )}
                      {uploadStatus === "success" && (
                        <svg className="w-20 h-20 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path className="checkmark-draw" strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                       {uploadStatus === "error" && (
                        <svg className="w-20 h-20 text-red-400 shake-simple" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      )}
                    </div>
                    
                    <div className="space-y-2 relative z-10">
                      <div className={`text-lg font-semibold transition-colors duration-300 ${uploadedFile ? "text-black" : "text-gray-500"}`}>
                        {uploadStatus === 'success' ? 'Upload complete' : 
                         uploadStatus === 'uploading' ? 'Uploading...' :
                         uploadedFile ? uploadedFile.name : "Choose a file"}
                      </div>
                      <div className={`text-sm transition-colors duration-300 ${uploadStatus === 'success' ? 'text-black' : 'text-gray-400 font-light'}`}>
                        {uploadStatus === 'success' ? 'Proceeding to training...' : 'or drag and drop'}
                      </div>
                    </div>
                  </div>
                </label>

                {uploadStatus === "error" && uploadError && (
                  <div className="animate-fadeIn glass-panel-error rounded-2xl p-4 text-red-600 text-sm font-medium text-center border-red-100">
                    {uploadError}
                  </div>
                )}

                {/* New Upload Results Accordion (Weather App Style) */}
                {uploadStatus === "success" && datasetInfo && (
                  <div className="animate-fadeIn space-y-4">
                    <button 
                      onClick={() => setUploadResultsExpanded(!uploadResultsExpanded)}
                      className="w-full glass-panel rounded-2xl p-4 flex items-center justify-between transition-all hover:bg-white/40 group"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="bg-blue-100 text-blue-700 p-2 rounded-lg">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                        </div>
                        <div className="text-left">
                          <div className="text-sm font-medium text-gray-500">Dataset Status</div>
                          <div className="text-black font-semibold">Ready for Training</div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-6">
                        <div className="text-right hidden sm:block">
                          <div className="text-xs text-gray-500 uppercase tracking-wide">Samples</div>
                          <div className="text-xl font-bold text-black">{datasetInfo.n_samples?.toLocaleString()}</div>
                        </div>
                        <svg className={`w-5 h-5 text-gray-400 transition-transform duration-300 ${uploadResultsExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                      </div>
                    </button>

                    <div className={`transition-all duration-500 ease-in-out overflow-hidden ${uploadResultsExpanded ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'}`}>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        {/* Features Tile */}
                        <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between h-24">
                          <div className="text-xs font-semibold text-gray-500 uppercase flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" /></svg>
                            Features
                          </div>
                          <div className="text-2xl font-bold text-black">{datasetInfo.n_features}</div>
                          <div className="text-xs text-gray-400">Input Columns</div>
                        </div>

                         {/* Memory Usage Tile */}
                         <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between h-24">
                          <div className="text-xs font-semibold text-gray-500 uppercase flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" /></svg>
                            Memory
                          </div>
                           <div className="text-2xl font-bold text-black">{datasetInfo.memory_usage_mb?.toFixed(2)} MB</div>
                          <div className="text-xs text-gray-400">RAM Usage</div>
                        </div>

                        {/* Target Range Tile (REPLACED Target Avg) */}
                        <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between h-24">
                          <div className="text-xs font-semibold text-gray-500 uppercase flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" /></svg>
                            Target Range
                          </div>
                          <div className="text-lg font-bold text-black truncate">
                            [{datasetInfo.feature_stats?.target_min.toFixed(1)}, {datasetInfo.feature_stats?.target_max.toFixed(1)}]
                          </div>
                          <div className="text-xs text-gray-400">Output Min-Max</div>
                        </div>
                      </div>

                      {/* REDESIGNED Feature Ranges Section */}
                      <div className="mt-6 space-y-4">
                        <div className="flex items-center space-x-2">
                          <div className="h-4 w-1 bg-black rounded-full"></div>
                          <div className="text-xs font-bold text-gray-500 uppercase tracking-wider">Feature Distribution (Pre-Standardization)</div>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                          {datasetInfo.feature_ranges?.map((range, idx) => (
                            <div key={idx} className="glass-panel rounded-2xl p-4 transition-all hover:scale-[1.01] duration-200">
                                <div className="flex justify-between items-center mb-3">
                                    <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Feature {idx + 1}</span>
                                </div>
                                
                                <div className="flex items-center space-x-3">
                                    <span className="text-sm font-mono font-semibold text-gray-600 w-12 text-right">{range.min.toFixed(1)}</span>
                                    
                                    {/* The Visual Bar */}
                                    <div className="flex-1 h-1.5 bg-gray-200/80 rounded-full overflow-hidden relative">
                                        <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 opacity-90"></div>
                                    </div>
                                    
                                    <span className="text-sm font-mono font-semibold text-gray-600 w-12 text-left">{range.max.toFixed(1)}</span>
                                </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <button
                  onClick={uploadStatus === "success" ? () => setCurrentStep(2) : handleUpload}
                  disabled={!uploadedFile || uploadStatus === "uploading"}
                  className="w-full bg-black text-white py-4 rounded-full text-base font-semibold hover:bg-gray-900 transition-all duration-300 shadow-sm disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  {uploadStatus === "uploading" ? "Processing..." : "Continue"}
                </button>
              </div>
            </div>
          )}

          {/* Step 2: Train */}
          {currentStep === 2 && (
            <div className="animate-fadeIn space-y-8">
              <div className="text-center pt-20">
                <h2 className="text-2xl font-semibold text-black mb-2 tracking-tight">Configure training</h2>
                <p className="text-sm text-gray-500 font-light">
                  Set your model parameters and begin training
                </p>
              </div>

              <div className="space-y-8">
                <div className="glass-panel rounded-3xl p-6 space-y-6">
                  {/* ... Inputs (Hidden Layers, Learning Rate, etc.) ... */}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-3">Hidden layers</label>
                    <div className="grid grid-cols-3 gap-3">
                      {trainingParams.hidden_sizes.map((size, idx) => (
                        <div key={idx} className="relative">
                          <input
                            type="number"
                            min="1"
                            max="1024"
                            value={size}
                            onChange={(e) => {
                              const value = parseInt(e.target.value);
                              if (isNaN(value)) return;
                              const newSizes = [...trainingParams.hidden_sizes];
                              newSizes[idx] = Math.max(1, Math.min(1024, value)); // Ensure between 1 and 1024
                              setTrainingParams({ ...trainingParams, hidden_sizes: newSizes });
                            }}
                            disabled={trainingStatus === "training" || trainingStatus === "success"}
                            className={`w-full px-4 py-3 bg-white/90 border-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-black/20 focus:border-black transition-all text-base text-black text-center font-medium ${
                              trainingStatus === "training" || trainingStatus === "success"
                                ? "border-gray-200 bg-gray-100/50 cursor-not-allowed opacity-60" 
                                : "border-gray-300"
                            }`}
                            placeholder={`Layer ${idx + 1}`}
                          />
                          {trainingParams.hidden_sizes.length > 1 && trainingStatus !== "training" && trainingStatus !== "success" && (
                            <button
                              onClick={() => {
                                const newSizes = trainingParams.hidden_sizes.filter((_, i) => i !== idx);
                                setTrainingParams({ ...trainingParams, hidden_sizes: newSizes });
                              }}
                              className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white rounded-full text-xs hover:bg-red-600 transition-colors flex items-center justify-center"
                            >
                              ×
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                    {trainingParams.hidden_sizes.length < 6 && trainingStatus !== "training" && trainingStatus !== "success" && (
                      <button
                        onClick={() => {
                          const lastSize = trainingParams.hidden_sizes[trainingParams.hidden_sizes.length - 1] || 32;
                          const newSizes = [...trainingParams.hidden_sizes, lastSize];
                          setTrainingParams({ ...trainingParams, hidden_sizes: newSizes });
                        }}
                        className="mt-2 text-xs text-gray-500 hover:text-gray-700 transition-colors flex items-center space-x-1"
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        <span>Add layer</span>
                      </button>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-semibold text-gray-700 mb-3">Learning rate</label>
                      <input
                        type="number"
                        step="0.0001"
                        min="0.0001"
                        max="1"
                        value={trainingParams.learning_rate}
                        onChange={(e) => {
                          const value = parseFloat(e.target.value);
                          if (isNaN(value)) return;
                          const clampedValue = Math.max(0.0001, Math.min(1, value));
                          setTrainingParams({
                            ...trainingParams,
                            learning_rate: clampedValue,
                          });
                        }}
                        disabled={trainingStatus === "training" || trainingStatus === "success"}
                        className={`w-full px-4 py-3 bg-white/90 border-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-black/20 focus:border-black transition-all text-base text-black font-medium ${
                          trainingStatus === "training" || trainingStatus === "success"
                            ? "border-gray-200 bg-gray-100/50 cursor-not-allowed opacity-60" 
                            : "border-gray-300"
                        }`}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-semibold text-gray-700 mb-3">Max iterations</label>
                      <input
                        type="number"
                        min="5"
                        max="1000"
                        value={trainingParams.max_iter}
                        onChange={(e) => {
                          const value = parseInt(e.target.value);
                          if (isNaN(value)) return;
                          const clampedValue = Math.max(5, Math.min(1000, value));
                          setTrainingParams({
                            ...trainingParams,
                            max_iter: clampedValue,
                          });
                        }}
                        disabled={trainingStatus === "training" || trainingStatus === "success"}
                        className={`w-full px-4 py-3 bg-white/90 border-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-black/20 focus:border-black transition-all text-base text-black font-medium ${
                          trainingStatus === "training" || trainingStatus === "success"
                            ? "border-gray-200 bg-gray-100/50 cursor-not-allowed opacity-60" 
                            : "border-gray-300"
                        }`}
                      />
                    </div>
                  </div>
                </div>

                {trainingStatus === "training" && (
                  <div className="glass-panel rounded-3xl p-6 space-y-4">
                    <div className="flex items-center justify-center space-x-3">
                      <svg className="animate-spin h-5 w-5 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span className="text-sm text-gray-700 font-medium">Training in progress</span>
                    </div>
                  </div>
                )}

                {trainingStatus === "error" && trainingError && (
                  <div className="glass-panel-error rounded-2xl p-5 text-red-600 text-sm font-medium text-center">
                    {trainingError}
                  </div>
                )}

                {trainingStatus === "success" && trainingResults && (
                  <div className="space-y-4">
                    {/* Collapsible Header Bar (Apple Weather Style) */}
                    <button 
                      onClick={() => setResultsExpanded(!resultsExpanded)}
                      className="w-full glass-panel rounded-2xl p-4 flex items-center justify-between transition-all hover:bg-white/40 group"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="bg-green-100 text-green-700 p-2 rounded-lg">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        </div>
                        <div className="text-left">
                          <div className="text-sm font-medium text-gray-500">Model Status</div>
                          <div className="text-black font-semibold">Training Complete</div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-6">
                        <div className="text-right hidden sm:block">
                          <div className="text-xs text-gray-500 uppercase tracking-wide">Test R²</div>
                          <div className="text-xl font-bold text-black">{trainingResults.test_r2?.toFixed(4)}</div>
                        </div>
                        <svg className={`w-5 h-5 text-gray-400 transition-transform duration-300 ${resultsExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                      </div>
                    </button>

                    {/* Expanded Dashboard */}
                    <div className={`transition-all duration-500 ease-in-out overflow-hidden ${resultsExpanded ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'}`}>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                        {/* Metrics Tiles */}
                        <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between h-28">
                          <div className="text-xs font-semibold text-gray-500 uppercase flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>
                            R² Score
                          </div>
                          <div className="text-2xl font-bold text-black">{trainingResults.val_r2?.toFixed(4)}</div>
                          <div className="text-xs text-gray-400">Validation</div>
                        </div>

                        <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between h-28">
                          <div className="text-xs font-semibold text-gray-500 uppercase flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" /></svg>
                            MSE
                          </div>
                          <div className="text-2xl font-bold text-black">{trainingResults.test_mse?.toFixed(4)}</div>
                          <div className="text-xs text-gray-400">Mean Sq. Error</div>
                        </div>

                         <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between h-28">
                          <div className="text-xs font-semibold text-gray-500 uppercase flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
                            RMSE
                          </div>
                          <div className="text-2xl font-bold text-black">{trainingResults.train_rmse?.toFixed(4)}</div>
                          <div className="text-xs text-gray-400">Root Mean Sq.</div>
                        </div>

                        <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between h-28">
                          <div className="text-xs font-semibold text-gray-500 uppercase flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            Time
                          </div>
                          <div className="text-2xl font-bold text-black">{trainingResults.training_time_seconds?.toFixed(2)}s</div>
                          <div className="text-xs text-gray-400">Duration</div>
                        </div>
                      </div>

                      {/* Charts Grid */}
                      <div className="grid grid-cols-1 gap-3">
                        {/* Loss Chart */}
                        <div className="glass-panel rounded-2xl p-4 h-64 relative">
                          <div className="text-xs font-semibold text-gray-500 uppercase mb-2 flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-blue-500"></span> Training Loss
                            <span className="w-2 h-2 rounded-full bg-orange-400 ml-2"></span> Validation Loss
                          </div>
                          <div className="h-[calc(100%-2.5rem)]">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={trainingResults.loss_history || []} margin={{ top: 5, right: 5, left: -25, bottom: -14 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                <XAxis dataKey="epoch" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                                <YAxis tick={{fontSize: 10}} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                                <Tooltip 
                                  contentStyle={{backgroundColor: 'rgba(255,255,255,0.8)', borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}}
                                  itemStyle={{fontSize: '12px'}}
                                />
                                <Line type="monotone" dataKey="loss" stroke="#3B82F6" strokeWidth={2} dot={false} />
                                <Line type="monotone" dataKey="val_loss" stroke="#FB923C" strokeWidth={2} dot={false} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>

                        {/* Predictions Scatter */}
                        <div className="glass-panel rounded-2xl p-4 h-64 relative">
                          <div className="text-xs font-semibold text-gray-500 uppercase mb-2">True (X) vs Predicted (Y) - Sample (up to 100 points)</div>
                          <div className="h-[calc(100%-2.5rem)]">
                            <ResponsiveContainer width="100%" height="100%">
                              <ScatterChart margin={{ top: 5, right: 5, left: -25, bottom: -14 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                                <XAxis type="number" dataKey="true" name="True" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                                <YAxis type="number" dataKey="pred" name="Predicted" tick={{fontSize: 10}} tickLine={false} axisLine={false} />
                                <ZAxis range={[60, 60]} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} 
                                  contentStyle={{backgroundColor: 'rgba(255,255,255,0.8)', borderRadius: '12px', border: 'none'}}
                                />
                                {trainingResults.predictions_sample && trainingResults.predictions_sample.length > 0 && (() => {
                                  const allVals = trainingResults.predictions_sample.flatMap(p => [p.true, p.pred]);
                                  const minVal = Math.min(...allVals);
                                  const maxVal = Math.max(...allVals);
                                  return (
                                    <ReferenceLine 
                                      segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]}
                                      stroke="#666" 
                                      strokeDasharray="3 3" 
                                      strokeWidth={1.5}
                                    />
                                  );
                                })()}
                                <Scatter name="Predictions" data={trainingResults.predictions_sample || []} fill="#000000" fillOpacity={0.6} />
                              </ScatterChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* THE REVERTED BUTTONS */}
                <div className="flex space-x-4">
                  <button
                    onClick={() => setCurrentStep(1)}
                    className="flex-1 bg-gray-100/80 backdrop-blur-xl text-black py-4 rounded-full text-base font-semibold hover:bg-gray-200/80 transition-all duration-200"
                  >
                    Back
                  </button>
                  <button
                    onClick={trainingStatus === "success" ? () => setCurrentStep(3) : handleTrain}
                    disabled={trainingStatus === "training"}
                    className="flex-1 bg-black text-white py-4 rounded-full text-base font-semibold hover:bg-gray-900 transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-sm"
                  >
                    <span>{trainingStatus === "training" ? "Training..." : trainingStatus === "success" ? "Continue" : "Train model"}</span>
                    {trainingStatus !== "training" && (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Predict */}
          {currentStep === 3 && (
            <div className="animate-fadeIn space-y-8">
              <div className="text-center pt-20">
                <h2 className="text-2xl font-semibold text-black mb-2 tracking-tight">Make a prediction</h2>
                <p className="text-sm text-gray-500 font-light">
                  Enter feature values to generate a prediction
                </p>
              </div>

              <div className="space-y-8">
                <div className="glass-panel rounded-3xl p-6 space-y-4">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <div key={i} className="flex items-center space-x-4">
                      <label className="w-24 text-sm font-semibold text-gray-700 text-right">
                        Feature {i + 1}
                      </label>
                      <input
                        type="number"
                        step="any"
                        value={features[i]}
                        onChange={(e) => updateFeature(i, e.target.value)}
                        className="flex-1 px-4 py-3 bg-white/90 border-2 border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-black/20 focus:border-black transition-all text-base text-center text-black font-medium"
                      />
                    </div>
                  ))}
                </div>

                {predictStatus === "error" && predictError && (
                  <div className="glass-panel-error rounded-2xl p-5 text-red-600 text-sm font-medium text-center">
                    {predictError}
                  </div>
                )}

                {predictStatus === "success" && prediction !== null && (
                  <div className="bg-black/95 backdrop-blur-xl text-white rounded-3xl p-12 text-center shadow-2xl">
                    <div className="text-xs font-semibold mb-4 opacity-60 uppercase tracking-wider">Prediction</div>
                    <div className="text-5xl font-semibold tracking-tight">{prediction.toFixed(4)}</div>
                  </div>
                )}

                <div className="flex space-x-4">
                  <button
                    onClick={() => setCurrentStep(2)}
                    className="flex-1 bg-gray-100/80 backdrop-blur-xl text-black py-4 rounded-full text-base font-semibold hover:bg-gray-200/80 transition-all duration-200"
                  >
                    Back
                  </button>
                  <button
                    onClick={handlePredict}
                    disabled={predictStatus === "predicting"}
                    className="flex-1 bg-black text-white py-4 rounded-full text-base font-semibold hover:bg-gray-900 transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed shadow-sm"
                  >
                    {predictStatus === "predicting" ? "Predicting..." : "Predict"}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }

        .mesh-gradient {
          position: absolute;
          width: 100%;
          height: 100%;
          background: linear-gradient(135deg, 
            #667eea 0%, 
            #764ba2 25%, 
            #f093fb 50%, 
            #4facfe 75%, 
            #00f2fe 100%);
          background-size: 400% 400%;
          animation: meshMove 40s ease infinite;
          filter: blur(100px);
        }

        @keyframes meshMove {
          0% { background-position: 0% 50%; }
          25% { background-position: 50% 100%; }
          50% { background-position: 100% 50%; }
          75% { background-position: 50% 0%; }
          100% { background-position: 0% 50%; }
        }

        .glass-panel {
          background: rgba(255, 255, 255, 0.7);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.3);
          box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        }

        .glass-panel-error {
          background: rgba(254, 242, 242, 0.9);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .arrow-rise {
          animation: riseUp 1.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        }

        @keyframes riseUp {
          0% { transform: translateY(4px); opacity: 0; }
          20% { opacity: 1; }
          80% { opacity: 1; }
          100% { transform: translateY(-8px); opacity: 0; }
        }

        .cloud-pulse {
          animation: subtlePulse 2s ease-in-out infinite;
        }

        @keyframes subtlePulse {
          0%, 100% { stroke-opacity: 1; }
          50% { stroke-opacity: 0.6; }
        }

        .checkmark-draw {
          stroke-dasharray: 30;
          stroke-dashoffset: 30;
          animation: drawStroke 0.6s cubic-bezier(0.65, 0, 0.45, 1) forwards;
        }

        @keyframes drawStroke {
          100% { stroke-dashoffset: 0; }
        }

        .shake-simple {
          animation: shakeSimple 0.4s ease-in-out both;
        }

        @keyframes shakeSimple {
          0%, 100% { transform: translateX(0); }
          20%, 60% { transform: translateX(-4px); }
          40%, 80% { transform: translateX(4px); }
        }
      `}</style>
    </div>
  );
}