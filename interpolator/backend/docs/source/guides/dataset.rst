Dataset Format Guide
====================

This guide explains the expected dataset format for the 5D Neural Network Interpolator.

Required Format
---------------

The system expects datasets in `.pkl` (pickle) format with the following structure:

.. code-block:: python

   {
       'X': numpy.ndarray,  # Shape: (n_samples, 5) - 5 feature columns
       'y': numpy.ndarray   # Shape: (n_samples,) - target values
   }

Requirements
------------

* **Exactly 5 features** (columns) in X
* X and y must have the **same number of samples**
* Missing values (NaN/inf) are automatically handled

Example: Creating a Dataset
----------------------------

Using the utility function:

.. code-block:: python

   from pydis_nn.utils import generate_sample_dataset
   import pickle
   
   # Generate synthetic dataset
   data = generate_sample_dataset(n=1000, seed=42)
   
   # Save to file
   with open('my_dataset.pkl', 'wb') as f:
       pickle.dump(data, f)

Creating from your own data:

.. code-block:: python

   import numpy as np
   import pickle
   
   # Your feature matrix (must have 5 columns)
   X = np.random.random((1000, 5))
   
   # Your target values
   y = np.random.random(1000)
   
   # Create dataset dictionary
   dataset = {
       'X': X.astype(np.float32),
       'y': y.astype(np.float32)
   }
   
   # Save to file
   with open('my_dataset.pkl', 'wb') as f:
       pickle.dump(dataset, f)

Data Validation
---------------

The system automatically validates:

* File format (must be .pkl)
* Dictionary structure (must have 'X' and 'y' keys)
* Shape compatibility (X and y must have matching sample counts)
* Feature count (X must have exactly 5 columns)

Missing Values
--------------

The system handles missing values automatically:

* **NaN/inf in features (X)**: Replaced with column mean
* **NaN/inf in targets (y)**: Samples are removed entirely

Preprocessing
-------------

During upload, the system:

1. Validates dataset format
2. Handles missing values
3. Computes dataset statistics (for display)
4. Stores the dataset for training

The actual preprocessing (splitting, standardization) happens during training.

Dataset Statistics
------------------

After upload, the system displays:

* Number of samples
* Number of features
* Missing values count
* Duplicate rows count
* Memory usage
* Feature statistics (min, max, mean, std)
* Feature ranges (per feature)
* Target statistics (min, max, mean, std)

