from easydict import EasyDict as edict

config = edict()
config.dataset = "emoreIresNet" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128 # batch size per GPU
config.lr = 0.1

# --- Updated Model Paths ---
config.backbone_weights = 'path/to/pretrained/backbone.pth' # e.g., './models/swin_cosface_backbone.pth'
config.header_weights = 'path/to/pretrained/header.pth'     # e.g., './models/swin_cosface_header.pth'
config.output = 'path/to/output/folder'                     # train model output folder
# ---------------------------

config.global_step = 0 # step to resume
config.s = 64.0
config.m = 0.5
config.std = 0.05
config.sched = 'cosine'
config.warmup_lr = 5e-4
config.min_lr = 5e-6
config.num_epoch =  10
config.warmup_epoch = 2
config.opt = 'adamw'
config.sparsity = 0.5
config.distillation = False
config.dist = 1
config.num_classes = 512

# type of network to train [iresnet100 | iresnet50| iresnet18]
config.network = "swin_transformer"
config.SE = False # SEModule
config.loss = "ArcFace"

if config.dataset == "emoreIresNet":
    # --- Updated Dataset Path ---
    config.rec = 'path/to/your/dataset/faces_emore' # e.g., './data/faces_emore'
    # ----------------------------
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  10
    config.warmup_epoch = 0
    config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 1686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14, 20, 25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func