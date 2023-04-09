export CUDA_VISIBLE_DEVICES=0,2
export NGPUS=2
export OMP_NUM_THREADS=8 # you can change this value according to your number of cpu cores


python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/tusimple.py
# python train.py configs/tusimple.py
