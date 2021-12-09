# Distributed-Systems-Project

This repository contains code the for the course project, which implements language modeling with support for synchronous and asynchrnous training on multiple GPUs. The dataset which is currently used in the code is based on the HateSpeech subset of the tweeteval dataset.

To run the synchronous training algorithm (say with 3 GPUs)

```python3 -m torch.distributed.launch --nproc_per_node 3 main_sync.py```

To run asynchronous training algorithm (say with 3 GPUs)

```python3 main_async.py --n_procs=3```

Each run will also print out the training dynamics (i.e. loss as the training progresses) in addition to a timestamp associated with every evaluation.

