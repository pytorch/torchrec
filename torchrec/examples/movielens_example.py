import torch

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.modules.embedding_modules import (
    EmbeddingBagConfig,
    EmbeddingBagCollection,
)

def main():
    # Dummy user interactions
    keys = ["movie_id"]
    values = torch.tensor([1, 4, 3, 5, 6], dtype=torch.long)
    lengths = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)  # 5 users, 1 movie each

    kjt = KeyedJaggedTensor(
        keys=keys,
        values=values,
        lengths=lengths,
    )

    ebc = EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(
                name="movie_id",
                embedding_dim=16,
                num_embeddings=1000,
            )
        ]
    )

    output = ebc(kjt)
    print("Output:", output["movie_id"].shape)  # Expected shape: (5, 16)

if __name__ == "__main__":
    main()
