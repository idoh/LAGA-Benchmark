export OMP_NUM_THREADS=6
echo 'Run training...'

algos=( "benchmark_ssgd_pytorch.py" "benchmark_laga_pytorch.py" )
networks=( "vgg16" "vgg19" "resnet18" "resnet50" "resnet101" "resnet152" "resnext101_32x8d" "wide_resnet101_2" ) 
ga=( 1 2 4 8 )

for a in "${algos[@]}"
do
    for n in "${networks[@]}"
    do
        for g in "${ga[@]}"
        do
            python -m torch.distributed.launch --nproc_per_node="$1" $a --batch-size=32 --batches-per-allreduce=$g --model=$n
        done
    done
done