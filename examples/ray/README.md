# Running torchrec with torchx using Ray scheduler on a Ray cluster

```
pip install --pre torchrec -f https://download.pytorch.org/whl/torchrec/index.html
pip install torchx-nightly
pip install "ray[default]" -qqq
```

Run torchx with the dashboard address and a link to your component
```
torchx run -s ray -cfg dashboard_address=localhost:6379,working_dir=~/repos/torchrec/examples/ray,requirements=./requirements.txt dist.ddp -j 1x2 --script ~/repos/torchrec/examples/ray/train_torchrec.py
```

Or run locally
```
torchx run -s ray -cfg working_dir=~/repos/torchrec/examples/ray,requirements=./requirements.txt dist.ddp -j 1x2 --script ~/repos/torchrec/examples/ray/train_torchrec.py
```

To run w/o ray scheduler (only torchx)
For available settings https://pytorch.org/torchx/latest/cli.html?highlight=torchx%20run
```
torchx run -s local_cwd dist.ddp -j 1x2 --script ~/repos/torchrec/examples/ray/train_torchrec.py
```

Job ID looks like ray://torchx/172.31.16.248:6379-raysubmit_ntquG1dDV6CtFUC5
Replace the job ID below by your string


Get a job status
PENDING, FAILED, INTERRUPTED ETC..
```
torchx status ray://torchx/172.31.16.248:6379-raysubmit_ntquG1dDV6CtFUC5
```

Get logs
```
torchx log ray://torchx/172.31.16.248:6379-raysubmit_ntquG1dDV6CtFUC5/worker/0
```
