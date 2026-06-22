import argparse
import logging
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
import timm
from utils import losses
from config.swin_config import config as cfg
from utils.dataset import MXFaceDataset
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from timm.scheduler import create_scheduler 
from pruning.swin_prune import SWIN_prune
from timm.optim import create_optimizer

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)


def main(args):


    # Set device to a single GPU
    device = torch.device("cuda:0")

    # Create output directory if it doesn't exist
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)

    # Initialize logging
    log_root = logging.getLogger()
    init_logging(log_root, 0, cfg.output)  # rank is 0 for single-GPU

    # Load dataset
    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=0)  # local_rank is 0 for single-GPU

    # Use a regular DataLoader
    train_loader = DataLoader(
        dataset=trainset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

 
    model=SWIN_prune(args=cfg, train_loader=None, test_loader=None)


    # Get header
    if cfg.loss == "ElasticArcFace":
        header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                      std=cfg.std).to(device)
    elif cfg.loss == "ElasticArcFacePlus":
        header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                      std=cfg.std, plus=True).to(device)
    elif cfg.loss == "ElasticCosFace":
        header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                      std=cfg.std).to(device)
    elif cfg.loss == "ElasticCosFacePlus":
        header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                      std=cfg.std, plus=True).to(device)
    elif cfg.loss == "ArcFace":
        header = losses.ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(device)

    elif cfg.loss == "CosFace":
        header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(device)
    else:
        print("Header not implemented")
        exit()


   #Optimizers
    opt_backbone = torch.optim.SGD(
        params=[{'params': model.model.parameters()}],
        lr=cfg.lr,  # Remove world_size since we're using a single GPU
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr,  # Remove world_size since we're using a single GPU
        momentum=0.9, weight_decay=cfg.weight_decay)

    opt_backbone = create_optimizer(model=model.model.parameters(), args=cfg)
    opt_header = create_optimizer(model=header.parameters(), args=cfg)

    # # Learning rate schedulers
    scheduler_backbone,_ = create_scheduler(optimizer=opt_backbone, args=cfg)
    scheduler_header,_=create_scheduler(optimizer=opt_header, args=cfg)
    
    # Loss function
    criterion = CrossEntropyLoss()
    # Training setup
    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)  # Remove world_size
    logging.info("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))

        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        # Update learning rates
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]
        print("last learning rate: {}".format(scheduler_header.get_lr()))

    # Callbacks
    callback_verification = CallBackVerification(cfg.eval_step, cfg.val_targets, cfg.rec)  # rank is 0
    callback_logging = CallBackLogging(
        frequent=50,
        total_step=total_step,
        batch_size=cfg.batch_size,
        writer=None
    )
    callback_checkpoint = CallBackModelCheckpoint(cfg.output)  # rank is 0

    # Load full-size model parameters
    model.load_model_parameters(cfg.backbone_weights)
    header.load_state_dict(torch.load(cfg.header_weights))

    logging.info(f'FULL MODEL PARAMETERS: {model.non_zero_el()/1000000:.2f} Million')

    model.compress('slimming', cfg.sparsity)

    logging.info(f'PRUNED MODEL PARAMETERS: {model.non_zero_el()/1000000:.2f} Million\n')

    # Save model flops
    logging.info(f'FULL MODEL FLOPS: {model.model.flops(sparsity=1)/1000000000:.2f} GFLOPs')
    logging.info(f'PRUNED MODEL FLOPS: {model.model.flops(sparsity=cfg.sparsity)/1000000000:.2f} GFLOPs\n')


    if cfg.distillation:
        teacher=SWIN_prune(args=cfg, train_loader=None, test_loader=None)
    
    # Training loop
    loss = AverageMeter()
    global_step = cfg.global_step

    for epoch in range(start_epoch, cfg.num_epoch):
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            _,_, x=model.model.forward(img)
            features = F.normalize(x)
            thetas = header(features, label)
            loss_v = criterion(thetas, label)
            if cfg.distillation:
                _,_,teacher_embeds = teacher.model(img)  
                teacher_embeds = F.normalize(teacher_embeds)
                loss_distill = 1 - F.cosine_similarity(features, teacher_embeds, dim=1).mean()
                full_loss=(1.-cfg.dist)*loss_v+cfg.dist*loss_distill
                full_loss.backward()
            else:
                loss_v.backward()
            clip_grad_norm_(model.model.parameters(), max_norm=5, norm_type=2)
            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            model.set_zero()
            callback_logging(global_step, loss, epoch) 
            callback_verification(global_step, model.model)


        scheduler_backbone.step(epoch)
        scheduler_header.step(epoch)

        callback_checkpoint(global_step, model.model, header)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine tunning pruned SWINFace model')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)