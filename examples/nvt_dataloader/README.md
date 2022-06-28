# Running torchrec using NVTabular DataLoader

First run nvtabular preprocessing to first convert the criteo TSV files to parquet, and perform offline preprocessing. For example
```
cd torchrec/torchrce/datasets/scripts/nvt/
python 01_nvt_preproc.py -i /data/criteo_1tb/ -o /data/criteo_1tb/
python 02_nvt_preproc.py -b /data/criteo_1tb/ --num_embeddings_per_feature 45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35

ls /data/criteo_1tb/criteo_preproc/*/*.parquet
>>> /data/criteo_1tb/criteo_preproc/day_0/part_0.parquet
>>> /data/criteo_1tb/criteo_preproc/day_1/part_0.parquet
```

To run locally
```
torchx run -s local_cwd dist.ddp -j 1x8 --script train_torchrec.py -- \
 --num_embeddings_per_feature 45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35 --over_arch_layer_sizes 1024,1024,512,256,1 --train_path /data/criteo_1tb/criteo_preproc/ --batch_size 4096
```
