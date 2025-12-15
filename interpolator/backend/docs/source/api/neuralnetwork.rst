Neural Network Module
======================

The ``pydis_nn.neuralnetwork`` module provides a configurable neural network implementation using TensorFlow/Keras.

.. automodule:: pydis_nn.neuralnetwork
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------

Basic usage:

.. code-block:: python

   from pydis_nn.neuralnetwork import NeuralNetwork
   from pydis_nn.data import load_and_preprocess
   
   # Load and preprocess data
   data = load_and_preprocess('dataset.pkl', random_state=42)
   
   # Create and train model
   model = NeuralNetwork(
       hidden_sizes=[64, 32, 16],
       learning_rate=0.001,
       max_iter=300,
       random_state=42
   )
   
   model.fit(
       data['X_train'],
       data['y_train'],
       X_val=data['X_val'],
       y_val=data['y_val']
   )
   
   # Make predictions
   predictions = model.predict(data['X_test'])
   
   # Evaluate
   r2_score = model.score(data['X_test'], data['y_test'])

