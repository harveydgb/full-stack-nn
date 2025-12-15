Utility Functions
=================

The ``pydis_nn.utils`` module provides utility functions for generating sample datasets.

.. automodule:: pydis_nn.utils
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Generate a synthetic dataset:

.. code-block:: python

   from pydis_nn.utils import generate_sample_dataset
   import pickle
   
   # Generate 1000 samples
   data = generate_sample_dataset(n=1000, seed=42)
   
   # Save to file
   with open('sample_dataset.pkl', 'wb') as f:
       pickle.dump(data, f)

