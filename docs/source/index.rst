.. TorchRec documentation master file, created by
   sphinx-quickstart on Fri Jan 14 11:37:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: TorchRec documentation homepage
   :keywords: recommendation systems, sharding, distributed training

Welcome to the TorchRec documentation!
======================================

TorchRec is a specialized library within the PyTorch ecosystem,
tailored for building, scaling, and deploying large-scale
**recommendation systems**, a niche not directly addressed by standard
PyTorch. TorchRec offers advanced features such as complex sharding
techniques for massive embedding tables, and enhanced distributed
training capabilities.

Getting Started
---------------

Topics in this section will help you get started with TorchRec.

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        TorchRec Overview
        :img-top: _static/img/card-background.svg
        :link: overview.html
        :link-type: url

        A short intro to TorchRec and why you need it.

     .. grid-item-card:: :octicon:`file-code;1em`
        Set up TorchRec
        :img-top: _static/img/card-background.svg
        :link: setup-torchrec.html
        :link-type: url

        Learn how to install and start using TorchRec
        in your environment.

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with TorchRec Tutorial
        :img-top: _static/img/card-background.svg
        :link: https://colab.research.google.com/github/pytorch/torchrec/blob/main/Torchrec_Introduction.ipynb
        :link-type: url

        Follow our interactive step-by-step tutorial
        to learn how to use TorchRec in a real-life
        example.



How to Contribute
-----------------

We welcome contributions and feedback from the PyTorch community!
If you are interested in helping improve the TorchRec project, here is
how you can contribute:

1. **Visit Our** `GitHub Repository <https://github.com/pytorch/torchrec>`__:
   There yoou can find the source code, issues, and ongoing projects.

1. **Submit Feedback or Issues**: If you encounter any bugs or have
   suggestions for improvements, please submit an issue through the
   `GitHub issue tracker <https://github.com/pytorch/torchrec/issues>`__.

1. **Propose changes**: Fork the repository and submit pull requests.
   Whether it's fixing a bug, adding new features, or improving
   documentation, your contributions are always welcome! Please make sure to
   review our `CONTRIBUTING.md <https://github.com/pytorch/torchrec/blob/main/CONTRIBUTING.md>`__

|
|

.. container:: center-content

   .. button-link:: https://github.com/pytorch/torchrec
      :color: info

      :octicon:`mark-github` Go to TorchRec Repo


.. toctree::
   :maxdepth: 1
   :caption: Introduction
   :hidden:

   overview.rst
   high_level_arch.rst

.. toctree::
   :maxdepth: 1
   :caption: All API References
   :hidden:

   torchrec.datasets.rst
   torchrec.datasets.scripts.rst
   torchrec.distributed.rst
   torchrec.distributed.planner.rst
   torchrec.distributed.sharding.rst
   torchrec.fx.rst
   torchrec.inference.rst
   torchrec.models.rst
   torchrec.modules.rst
   torchrec.optim.rst
   torchrec.quant.rst
   torchrec.sparse.rst
   torchrec.metrics.rst
