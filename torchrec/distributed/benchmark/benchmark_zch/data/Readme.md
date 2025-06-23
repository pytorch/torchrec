# Datasets for zero collision hash benchmark

## Folder structure
- `configs/`: Configs for each dataset, named as `{dataset_name}.json`
- `preprocess`: Include scripts to preprocess the dataset to make the returned dataset in the format of
    - batch
        - dense_features
        - sparse_features
        - labels
- `get_dataloader.py`: the entry point to get the dataloader for each dataset
