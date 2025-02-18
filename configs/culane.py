# DATA
dataset = 'CULane'
data_root = '/data/ldp/zjf/dataset/CULane'

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD'  # ['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi'  # ['multi', 'cos']
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '/data/ldp/zjf/log/CULane'

# FINETUNE or RESUME MODEL PATH
finetune = None
# finetune = '/data/ldp/zjf/log/Tusimple/20230407_160210_lr_4e-04_b_32/ep399.pth'
resume = None

# TEST and Visualization
# model_mark = 模型所属的训练/在该训练中所属的批次【即相对log_path的路径】
model_mark = '/20221219_161349_lr_1e-01_b_32/ep049.pth'
test_model = '/data/ldp/zjf/log/CULane' + model_mark
# test_work_dir = '/data/ldp/zjf/evaluation/ep049' + model_mark
visualize_work_dir = '/data/ldp/zjf/visualization/ep049' + model_mark
test_work_dir = '/data/ldp/zjf/evaluation2/CULane' + model_mark
# visualize_work_dir = '/data/ldp/zjf/visualization/CULane' + model_mark

num_lanes = 4
