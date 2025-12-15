Data Handling Module
====================

The ``pydis_nn.data`` module provides functionality for loading, validating, and preprocessing 5D datasets.

.. automodule:: pydis_nn.data
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Loading a dataset:

.. code-block:: python

   from pydis_nn.data import load_dataset
   
   data = load_dataset('my_dataset.pkl')
   X, y = data['X'], data['y']

Complete preprocessing pipeline:

.. code-block:: python

   from pydis_nn.data import load_and_preprocess
   
   data = load_and_preprocess(
       'my_dataset.pkl',
       train_size=0.7,
       val_size=0.15,
       test_size=0.15,
       standardize=True,
       random_state=42
   )
   
   X_train = data['X_train']
   y_train = data['y_train']

