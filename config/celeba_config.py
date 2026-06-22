from easydict import EasyDict as edict

config = edict()
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64 # batch size per GPU
config.lr = 0.01
config.output = "output/iresnet18_celeba" # train model output folder
config.global_step=0 # step to resume
config.s=64.0
config.m=0.5
config.std=0.05
config.num_epoch = 20
config.num_classes=40
config.SE=False
