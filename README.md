# LAGA-Benchmark
This repository benchmarks a variety of neural network architecture on the ImageNet dataset.

The code is based on [ImageNet example in Horovod](https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_synthetic_benchmark.py).

## Dependencies
- [Horovod Docker](https://hub.docker.com/layers/horovod/horovod/0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1/images/sha256-e7459ce7e799b09cb4da463d6e997e8a50212972d7bfac2c218f9080a2c8e24b?context=explore)
- [Pandas](https://pypi.org/project/pandas/)

## Recipes
Inside the docker image run:
```
bash benchmark_pytorch.sh --ngpus [#GPUs]
```

## Benchmark Performance Results
The results can be found in the `results` folder as `csv` files.

## Citation
If you find this useful, please cite our work as:

Ido Hakimi, Rotem Zamir Aviv, Kfir Y. Levy, and Assaf Schuster. LAGA: Lagged AllReduce with Gradient Accumulation for Minimal Idle Time. In Proc. of ICDM, 2021.

```
@inproceedings{hakimi2021laga,
  title={LAGA: Lagged AllReduce with Gradient Accumulation for Minimal Idle Time},
  author={Hakimi, Ido and Zamir Aviv, Rotem and Levy, Kfir Yehuda and Schuster, Assaf},
  booktitle = {2021 IEEE International Conference on Data Mining (ICDM)},
  year={2021}
}
```
