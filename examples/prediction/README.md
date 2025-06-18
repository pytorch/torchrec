# DLRM Prediction Example

This example demonstrates how to use a Deep Learning Recommendation Model (DLRM) for making predictions using TorchRec capabilities. The code includes:

1. A DLRM implementation using TorchRec's EmbeddingBagCollection and KeyedJaggedTensor
2. Training with random data
3. Evaluation
4. Making sample predictions

## TorchRec Integration

This implementation has been updated to use TorchRec's capabilities:
- Uses `KeyedJaggedTensor` for sparse features
- Uses `EmbeddingBagCollection` for embedding tables
- Follows the DLRM architecture as described in the paper: https://arxiv.org/abs/1906.00091

The example demonstrates how to leverage TorchRec's efficient sparse feature handling for recommendation models.

## Dependencies

Install the required dependencies:

```bash
# Install PyTorch
pip install torch torchvision

# Install NumPy
pip install numpy

# Install TorchRec
pip install torchrec
```

**Important**: This implementation now requires torchrec to run, as it uses TorchRec's specialized modules for recommendation systems.

## Running the Example Locally

1. Download the `predict_using_torchrec.py` file to your local machine.

2. Run the example:

```bash
python3 predict_using_torchrec.py
```

3. If you're using a different Python environment:

```bash
# For conda environments
conda activate your_environment_name
python predict_using_torchrec.py

# For virtual environments
source your_venv/bin/activate
python predict_using_torchrec.py
```

## What to Expect

When you run the example, you'll see:

1. Training progress for 10 epochs with loss and learning rate information
2. Evaluation results showing MSE and RMSE metrics
3. Sample predictions for a specific user on multiple items

## Implementation Details

This example uses TorchRec's capabilities to implement a DLRM model that:

- Takes dense features and sparse features (as KeyedJaggedTensor) as input
- Processes dense features through a bottom MLP
- Processes sparse features through EmbeddingBagCollection
- Computes feature interactions using dot products
- Processes the interactions through a top MLP
- Outputs rating predictions on a 0-5 scale

The implementation demonstrates how to use TorchRec's specialized modules for recommendation systems, making it more efficient and scalable than a custom implementation.

## Key TorchRec Components Used

1. **KeyedJaggedTensor**: Efficiently represents sparse features with variable lengths
2. **EmbeddingBagConfig**: Configures embedding tables with parameters like dimensions and feature names
3. **EmbeddingBagCollection**: Manages multiple embedding tables for different categorical features

## Troubleshooting

If you encounter any issues:

1. **Python version**: This code has been tested with Python 3.8+. Make sure you're using a compatible version.

2. **PyTorch and TorchRec installation**: If you have issues with PyTorch or TorchRec, try installing specific versions:
   ```bash
   pip install torch==2.0.0 torchvision==0.15.0
   pip install torchrec==0.5.0
   ```

3. **Memory issues**: If you run out of memory, try reducing the batch size by modifying this line in the code:
   ```python
   batch_size = 256  # Try a smaller value like 64 or 32
   ```

4. **CPU vs GPU**: The code automatically uses CUDA if available. To force CPU usage, modify:
   ```python
   device = torch.device("cpu")
   ```

5. **TorchRec compatibility**: If you encounter compatibility issues with TorchRec, make sure you're using compatible versions of PyTorch and TorchRec.
