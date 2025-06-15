# Two-Tower Model for Efficient Retrieval

This example demonstrates how to use a Two-Tower model for efficient retrieval in large-scale recommendation systems using TorchRec capabilities. The code includes:

1. A Two-Tower implementation using TorchRec's EmbeddingBagCollection and KeyedJaggedTensor
2. Training with random data
3. Evaluation
4. Making sample predictions
5. Separate methods for generating user and item embeddings for retrieval

## Two-Tower Architecture for Retrieval

The Two-Tower model is a powerful architecture for large-scale retrieval in recommendation systems, as described in [Google Cloud's article on scaling deep retrieval](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture). It consists of:

- **User/Query Tower**: Processes user features to create user embeddings
- **Item Tower**: Processes item features to create item embeddings
- **Shared Embedding Space**: Both towers project features into the same semantic space
- **Similarity Calculation**: Typically dot product or cosine similarity between embeddings

### Key Advantages for Large-Scale Retrieval

1. **Computational Efficiency**: Item embeddings can be pre-computed offline and indexed
2. **Serving Efficiency**: Only the user tower needs to be computed at serving time
3. **Scalability**: Avoids scoring all items by using approximate nearest neighbor search
4. **Separation of Concerns**: Each tower can be optimized independently

## TorchRec Integration

This implementation uses TorchRec's capabilities:
- Uses `KeyedJaggedTensor` for sparse features
- Uses `EmbeddingBagCollection` for embedding tables
- Implements the Two-Tower architecture as described in Google Cloud's article on scaling deep retrieval
- Provides methods for generating embeddings for retrieval use cases

The example demonstrates how to leverage TorchRec's efficient sparse feature handling for large-scale retrieval systems.

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

**Important**: This implementation requires torchrec to run, as it uses TorchRec's specialized modules for recommendation systems.

## Running the Example Locally

1. Download the `predict_using_twotower.py` file to your local machine.

2. Run the example:

```bash
python3 predict_using_twotower.py
```

3. If you're using a different Python environment:

```bash
# For conda environments
conda activate your_environment_name
python predict_using_twotower.py

# For virtual environments
source your_venv/bin/activate
python predict_using_twotower.py
```

## What to Expect

When you run the example, you'll see:

1. Training progress for 10 epochs with loss and learning rate information
2. Evaluation results showing MSE and RMSE metrics
3. Sample predictions for specific user-item pairs

## Implementation Details

This example uses TorchRec's capabilities to implement a Two-Tower model that:

- Takes user and item IDs as input (as KeyedJaggedTensor)
- Processes user IDs through the user tower (EmbeddingBagCollection + MLP)
- Processes item IDs through the item tower (EmbeddingBagCollection + MLP)
- Projects both into a shared embedding space
- Computes relevance scores using dot products between user and item embeddings
- Provides methods for generating embeddings for retrieval use cases:
  - `get_user_embedding()`: For online serving
  - `get_item_embedding()`: For offline indexing
- Supports optional L2 normalization for cosine similarity

### Retrieval Workflow

In a real-world retrieval system, you would:

1. **Offline Processing**:
   - Pre-compute embeddings for all items using `get_item_embedding()`
   - Index these embeddings in an approximate nearest neighbor (ANN) system like FAISS

2. **Online Serving**:
   - Compute the user embedding using `get_user_embedding()`
   - Query the ANN index to find the most similar items
   - Optionally re-rank the top candidates with a more complex model

This approach scales to millions or billions of items by avoiding the need to score all items for each user query.

## Key TorchRec Components Used

1. **KeyedJaggedTensor**: Efficiently represents sparse features with variable lengths
2. **EmbeddingBagConfig**: Configures embedding tables with parameters like dimensions and feature names
3. **EmbeddingBagCollection**: Manages embedding tables for user and item features
4. **MLP**: Implements the tower architectures for processing embeddings

## Comparison with DLRM

While both DLRM and Two-Tower models are used for recommendation systems, they have different architectures and use cases:

- **DLRM**:
  - Combines multiple categorical features and dense features with feature interactions
  - Designed for CTR prediction and ranking tasks
  - Scores each user-item pair individually
  - Better for final ranking of a small set of candidates

- **Two-Tower**:
  - Separates user and item processing into distinct towers
  - Designed specifically for efficient retrieval at scale
  - Enables pre-computation of item embeddings
  - Allows approximate nearest neighbor search
  - Better for retrieving relevant candidates from a large catalog

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
