Inference
----------------------------------

TorchRec provides easy-to-use APIs for transforming an authored TorchRec model
into an optimized inference model for distributed inference, via eager module swaps.

This transforms TorchRec modules like ``EmbeddingBagCollection`` in the model to
a quantized, sharded version that can be compiled using torch.fx and TorchScript
for inference in a C++ environment.

The intended use is calling ``quantize_inference_model`` on the model followed by
``shard_quant_model``.

.. codeblock::

.. automodule:: torchrec.inference.modules

.. autofunction:: quantize_inference_model
.. autofunction:: shard_quant_model
