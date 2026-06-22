from easydict import EasyDict as edict

config = edict()
config.momentum=0.9
config.embedding_size = 512 # embedding size of model
config.weight_decay = 0.01
config.batch_size = 16 # batch size per GPU
config.lr = 0.3e-4
config.data_path='../Bias_testing/data/celeba'
config.output = "../Bias_testing/output/swin_celeba" # train model output folder
config.global_step=0 # step to resume
config.s=64.0
config.m=0.5
config.std=0.05
config.sched='cosine'
config.warmup_lr = 5e-5
config.min_lr = 5e-6
config.num_epoch =  10
config.warmup_epoch = 2
config.opt='adamw'
config.num_classes=40

