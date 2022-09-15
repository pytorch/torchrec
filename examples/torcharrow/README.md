# Description

This shows a prototype of integrating a TorchRec based training loop utilizing TorchArrow's on-the-fly preprocessing. The main motivation is to show the utilization of TorchArrow's specialized domain UDFs. Here we use `bucketize`, `firstx`, as well as `sigrid_hash` to do some last-mile preprocessing over the criteo dataset in parquet format. More recommendation domain functions can be found at [torcharrow.functional Doc](https://pytorch.org/torcharrow/beta/functional.html#recommendation-operations).

These three UDFs are extensively used in Meta's RecSys preprocessing stack. Notably, these UDFs can be used to easily adjust the proprocessing script to any model changes. For example, if we wish to change the size of our embedding tables, without sigrid_hash, we would need to rerun a bulk offline preproc to ensure that all indicies are within bounds. Bucketize lets us easily convert dense features into sparse features, with flexibility of what the bucket borders are. firstx lets us easily prune sparse ids (note, that this doesn't provide any functional features, but is in the preproc script as demonstration).


## Installations and Usage

Download the criteo tsv files (see the README in the main DLRM example). Use the nvtabular script (in torchrec/datasets/scripts/nvt/) to convert the TSV files to parquet.

To start, install torcharrow-nightly and torchdata:
```
pip install --pre torcharrow -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
pip install torchdata
```
You can also build TorchArrow from source, following https://github.com/pytorch/torcharrow

Usage

```
torchx run -s local_cwd dist.ddp -j 1x4 --script examples/torcharrow/run.py -- --parquet_directory /home/criteo_parquet
```

The preprocessing logic is in ```dataloader.py```

## Extentions/Future work

* We will eventually integrate with the up and coming DataLoader2, which will allow us to utilize a prebuilt solution to collate our dataframe batches to dense tensors, or TorchRec's KeyedJaggedTensors (rather than doing this by hand).
* Building an easier solution/more performant to convert parquet -> IterableDataPipe[torcharrow.DataFrame] (aka ArrowDataPipe). Also currently batch sizes are not available.
* Some functional abilities are not yet available (such as make_named_row, etc).
* Support collation/conversion for ArrowDataPipe
* More RecSys UDFs to come! Please let us know if you have any suggestions.
