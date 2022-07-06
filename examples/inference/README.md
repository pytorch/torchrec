# Examples

Note - please use torchrec_nightly >= 2022.6.30 for quant embedding_bag_collection optimizations

## Package DLRM Model for inference

`Torchrec.inference` provides utilities to package models using `torch.package`.

`torch.package`, as an alternative to TorchScript, lets you package your Python model code. `torch::deploy` interpreters can subsequently use this model code from C++ to create models. The advantage is that `torch.package` enables you to package model code that includes TorchRec sharding, which is not possible with TorchScript. This is the crux behind GPU inference. Read more about:
- `torch.package` - https://pytorch.org/docs/stable/package.html\
- `torch::deploy` - https://github.com/pytorch/pytorch/blob/master/docs/source/deploy.rst


### **PredictModule and PredictFactory** - Model code for inference

The `DLRMPredictFactory` class (implements `PredictFactory` from `torchrec.inference.modules`) in `dlrm_predict.py` describes the model architecture that will be used for inference. The `torch::deploy` interpreters will use the `create_predict_module` method to create the `DLRMPredictModule` (implements `PredictModule` from `torchrec.inference.modules`).

This model includes sharding from `TorchRec` as it is wrapped in `DistributedModelParallel`.

### Unsharded example - single GPU

In order to use the single GPU example with no sharding from TorchRec, replace `DLRMPredictFactory` with `DLRMPredictSingleGPUFactory` (from `dlrm_predict_single_gpu.py`) in `dlrm_packager.py`.

### **PredictFactoryPackager** - Packaging the model code

The `dlrm_packager.py` script can be used to package the model code:

```
python dlrm_packager.py --output_path /tmp/model_package.zip
```

The `DLRMPredictFactoryPackager` class (implements `PredictFactoryPackager` from `torchrec.inference.model_packager`) details which dependencies of the model are "external". This lets the deploy interpreters know to look for these dependencies in the system's Python packages. Read more about external dependencies to packages in the documentation for `torch.package` listed above.


## Run client for prediction requests

Once the `torchrec.inference` [server](https://github.com/pytorch/torchrec/blob/main/torchrec/inference/server.cpp) is running, run `dlrm_client.py` to send prediction requests for the dlrm model.

```
python dlrm_client.py
```

Receive response back:

```
Response:  [0.13199582695960999, -0.1048036441206932, -0.06022112816572189, -0.08765199035406113, -0.12735335528850555, -0.1004377081990242, 0.05509107559919357, -0.10504599660634995, 0.1350800096988678, -0.09468207508325577, 0.24013587832450867, -0.09682435542345047, 0.0025023818016052246, -0.09786031395196915, -0.26396819949150085, -0.09670191258192062, 0.2691854238510132, -0.10246685892343521, -0.2019493579864502, -0.09904996305704117, 0.3894067406654358, ...]
```
