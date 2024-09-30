.. _overview_label:

==================
TorchRec Overview
==================

TorchRec is the PyTorch recommendation system library, designed to provide common primitives
for creating state-of-the-art personalization models and a path to production. TorchRec is
widely adopted in many Meta production recommendation system models for training and inference workflows.

Why TorchRec?
------------------

TorchRec is designed to address the unique challenges of building, scaling and deploying massive,
large-scale recommendation system models, which is not a focus of regular PyTorch. More specifically,
TorchRec provides the following primitives for general recommendation systems:

- **Specialized Components**: TorchRec provides simplistic, specialized modules that are common in authoring recommendation systems, with a focus on embedding tables
- **Advanced Sharding Techniques**: TorchRec provides flexible and customizable methods for sharding massive embedding tables: Row-Wise, Column-Wise, Table-Wise, and so on. TorchRec can automatically determine the best plan for a device topology for efficient training and memory balance
- **Distributed Training**: While PyTorch supports basic distributed training, TorchRec extends these capabilities with more sophisticated model parallelism techniques specifically designed for the massive scale of recommendation systems
- **Incredibly Optimized**: TorchRec training and inference components are incredibly optimized on top of FBGEMM. After all, TorchRec powers some of the largest recommendation system models at Meta
- **Frictionless Path to Deployment**: TorchRec provides simple APIs for transforming a trained model for inference and loading it into a C++ environment for the most optimal inference model
- **Integration with PyTorch Ecosystem**: TorchRec is built on top of PyTorch, meaning it integrates seamlessly with existing PyTorch code, tools, and workflows. This allows developers to leverage their existing knowledge and codebase while utilizing advanced features for recommendation systems. By being a part of the PyTorch ecosystem, TorchRec benefits from the robust community support, continuous updates, and improvements that come with PyTorch.
