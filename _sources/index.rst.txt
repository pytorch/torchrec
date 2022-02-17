.. TorchRec documentation master file, created by
   sphinx-quickstart on Fri Jan 14 11:37:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the TorchRec documentation!
======================================

TorchRec is a PyTorch domain library built to provide common
sparsity & parallelism primitives needed for large-scale recommender
systems (RecSys). It allows authors to train models with large
embedding tables sharded across many GPUs.

For installation instructions, visit

https://github.com/pytorch/torchrec#readme



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   torchrec.datasets.rst
   torchrec.distributed.rst
   torchrec.fx.rst
   torchrec.models.rst
   torchrec.modules.rst
   torchrec.optim.rst
   torchrec.quant.rst
   torchrec.sparse.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
