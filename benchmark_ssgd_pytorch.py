import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

from torchvision import models
import timeit
import math
import pandas as pd
import numpy as np

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=50,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=25,
                    help='number of batches per benchmark iteration')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--local_rank',  type=int,
                    default=os.getenv('LOCAL_RANK', 0),
                    help='Used for multi-process training.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.distributed.init_process_group(backend="nccl", init_method='env://')
assert torch.distributed.is_initialized()

world_size = torch.distributed.get_world_size()
rank = torch.distributed.get_rank()

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(args.local_rank)

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()

# By default, Adasum doesn't need scaling up learning rate.
lr_scaler = world_size

if args.cuda:
    # Move model to GPU.
    model.cuda()
    model = DistributedDataParallel(model,
                            device_ids=[args.local_rank],
                            output_device=args.local_rank,
                            broadcast_buffers=False,
                            find_unused_parameters=True,
                            )

optimizer = optim.SGD(model.parameters(), lr=0.01 * lr_scaler)

# Set up fixed fake data
allreduce_batch_size = args.batch_size * args.batches_per_allreduce
data = torch.randn(allreduce_batch_size, 3, 224, 224)
target = torch.LongTensor(allreduce_batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()

def benchmark_step():
    optimizer.zero_grad()
    for i in range(0, len(data), args.batch_size):
        data_batch = data[i:i + args.batch_size]
        target_batch = target[i:i + args.batch_size]
        if i + args.batch_size >= len(data) - 1:
            output = model(data_batch)
            loss = F.cross_entropy(output, target_batch)
            # Average gradients among sub-batches
            loss.div_(math.ceil(float(len(data)) / args.batch_size))
            loss.backward()
        else:
            with model.no_sync():
                output = model(data_batch)
                loss = F.cross_entropy(output, target_batch)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()

    optimizer.step()


def log(s, nl=True):
    if rank != 0:
        return
    print(s, end='\n' if nl else '')


log('PyTorch - SSGD')
log('Model: %s' % args.model)
log(f'AllReduce size: {allreduce_batch_size * world_size}')
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, world_size))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = allreduce_batch_size * args.num_batches_per_iter / time
    # log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (world_size, device, world_size * img_sec_mean, world_size * img_sec_conf))


if rank == 0:
    name = "pytorch_ssgd_tmp"
    path = f"results/{name}.csv"

    def create_row(img_sec):
        return {
            "Algorithm": name,
            "Architecture": args.model,
            "Accumulations": args.batches_per_allreduce,
            "Micro-Batch Size": args.batch_size,
            "Images/Second": world_size * img_sec,
        }

    rows = [create_row(img_sec) for img_sec in img_secs]

    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    df = df.append(rows, ignore_index=True)
    df.to_csv(path, index=False)