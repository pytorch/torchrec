# TorchRec Benchmark
## usage
- internal:
```
hash=$(hg whereami | cut -c 1-10)
buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_train_pipeline -- \
    --yaml_config=fbcode/torchrec/distributed/benchmark/yaml/sparse_data_dist_base.yml \
    --profile_name=sparse_data_dist_base_${hash:-$USER} # overrides the yaml config
```
- oss:
```
hash=`git rev-parse --short HEAD`
python -m torchrec.distributed.benchmark.benchmark_train_pipeline \
    --yaml_config=fbcode/torchrec/distributed/benchmark/yaml/sparse_data_dist_base.yml \
    --profile_name=sparse_data_dist_base_${hash:-$USER} # overrides the yaml config
```
