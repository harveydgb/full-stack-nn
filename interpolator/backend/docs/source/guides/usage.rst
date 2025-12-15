Usage Guide
===========

This guide demonstrates how to use the 5D Neural Network Interpolator system.

Web Interface Workflow
----------------------

1. **Upload Dataset**
   
   * Open http://localhost:3000 in your browser
   * Click "Choose a file" and select your .pkl dataset file
   * Click "Continue" - the system will validate and display dataset statistics

2. **Train Model**
   
   * Configure model parameters:
     * Hidden layer sizes (default: [64, 32, 16])
     * Learning rate (default: 0.001)
     * Max iterations (default: 300)
   * Click "Train model"
   * View training metrics and graphs

3. **Make Predictions**
   
   * Enter 5 feature values
   * Click "Predict"
   * View the prediction result

Python API Usage
----------------

Direct Library Usage
~~~~~~~~~~~~~~~~~~~~

You can also use the Python modules directly:

.. code-block:: python

   from pydis_nn.data import load_and_preprocess
   from pydis_nn.neuralnetwork import NeuralNetwork
   from pydis_nn.utils import generate_sample_dataset
   
   # Generate or load dataset
   data = generate_sample_dataset(n=1000, seed=42)
   
   # Or load from file
   # from pydis_nn.data import load_and_preprocess
   # data = load_and_preprocess('my_dataset.pkl', random_state=42)
   
   # Create model
   model = NeuralNetwork(
       hidden_sizes=[64, 32, 16],
       learning_rate=0.001,
       max_iter=300,
       random_state=42
   )
   
   # Train
   model.fit(
       data['X_train'],
       data['y_train'],
       X_val=data['X_val'],
       y_val=data['y_val']
   )
   
   # Evaluate
   test_r2 = model.score(data['X_test'], data['y_test'])
   print(f"Test R²: {test_r2:.4f}")
   
   # Predict
   predictions = model.predict(data['X_test'])

REST API Usage
~~~~~~~~~~~~~~

Using curl:

Upload dataset:

.. code-block:: bash

   curl -X POST "http://localhost:8000/upload" \
        -F "file=@my_dataset.pkl"

Train model:

.. code-block:: bash

   curl -X POST "http://localhost:8000/train" \
        -H "Content-Type: application/json" \
        -d '{
          "hidden_sizes": [64, 32, 16],
          "learning_rate": 0.001,
          "max_iter": 300,
          "random_state": 42
        }'

Make prediction:

.. code-block:: bash

   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "features": [0.1, 0.2, 0.3, 0.4, 0.5]
        }'

Using Python requests:

.. code-block:: python

   import requests
   
   # Upload dataset
   with open('my_dataset.pkl', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/upload',
           files={'file': f}
       )
   
   # Train model
   response = requests.post(
       'http://localhost:8000/train',
       json={
           'hidden_sizes': [64, 32, 16],
           'learning_rate': 0.001,
           'max_iter': 300
       }
   )
   results = response.json()
   print(f"Test R²: {results['test_r2']:.4f}")
   
   # Make prediction
   response = requests.post(
       'http://localhost:8000/predict',
       json={'features': [0.1, 0.2, 0.3, 0.4, 0.5]}
   )
   prediction = response.json()['prediction']
   print(f"Prediction: {prediction}")

