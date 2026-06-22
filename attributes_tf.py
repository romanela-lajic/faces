import argparse
import logging
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import BCEWithLogitsLoss

import timm
from utils import losses
from config.celeba_tf_config import config as cfg
from utils.load_celeba import full_celeba_get_datasets
from utils.utils_callbacks import CallBackModelCheckpoint
from utils.utils_logging import AverageMeter
from pruning.swin_prune import SWIN_prune
from timm.scheduler import create_scheduler 
from timm.optim import create_optimizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from tqdm import tqdm                                                                 

# Setup logging
def setup_logger(log_dir, training=True):
    os.makedirs(log_dir, exist_ok=True)
    if training:
        log_path = os.path.join(log_dir, "training.log")
    else:
        log_path = os.path.join(log_dir, "test.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def celeba_classes():
    return ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

@torch.no_grad()
def validate(backbone, header, val_loader, device, logger):
    backbone.eval()
    header.eval()

    all_preds = []
    all_targets = []
    total_correct = 0
    total_labels = 0

    for imgs, labels in tqdm(val_loader, desc="Validating", ncols=100):
        imgs = imgs.to(device)
        labels = labels.to(device)

        _, _, x = backbone(imgs)
        features = F.normalize(x)
        logits = header(features)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu().int())

        correct = (preds == labels).sum().item()
        total_correct += correct
        total_labels += labels.numel()

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    mean_accuracy = total_correct / total_labels

    logger.info(f" Validation Metrics:")
    logger.info(f"   Mean Accuracy : {mean_accuracy:.4f}")
    logger.info(f"   Precision     : {precision:.4f}")
    logger.info(f"   Recall        : {recall:.4f}")
    logger.info(f"   F1 Score      : {f1:.4f}")

    backbone.train()
    header.train()

def train(cfg, seed, sparsity):
        torch.manual_seed(seed)
        device = torch.device("cuda:0")
        output=cfg.output+'_'+str(seed)+'_'+str(sparsity)
        input=cfg.output+'_'+str(seed)
        input_backbone=os.path.join(input, '25420backbone.pth')
        input_header=os.path.join(input, '25420header.pth')
        os.makedirs(output, exist_ok=True)
        logger = setup_logger(output)

        logger.info("Initializing datasets...")
        trainset, valid_set, test_set = full_celeba_get_datasets('../Bias_testing/data/celeba', return_test=True)
        train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        logging.info(f'RANDOM SEED: {seed}\n')
        logging.info(f'SPARSITY: {sparsity}\n')
        model = SWIN_prune(cfg, None, None)
        header = losses.MultiLabelHeader(in_features=cfg.embedding_size, out_features=40).to(device)
        model.model.load_state_dict(torch.load(input_backbone))
        model.compress('slimming', sparsity)
        header.load_state_dict(torch.load(input_header))
        header.train()
        model.model.train()

        opt_backbone = create_optimizer(model=model.model.parameters(), args=cfg)
        opt_header = create_optimizer(model=header.parameters(), args=cfg)
        scheduler_backbone,_ = create_scheduler(optimizer=opt_backbone, args=cfg)
        scheduler_header,_=create_scheduler(optimizer=opt_backbone, args=cfg)

        criterion = BCEWithLogitsLoss()
        start_epoch = 0
        global_step = cfg.global_step
        total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)
        logger.info(f"Total training steps: {total_step}")


        callback_checkpoint = CallBackModelCheckpoint(output)

        loss_meter = AverageMeter()

        for epoch in range(start_epoch, cfg.num_epoch):
            logger.info(f"\nStarting Epoch [{epoch+1}/{cfg.num_epoch}]")
            loss_meter.reset()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", ncols=100)

            for step, (img, label) in pbar:
                global_step += 1

                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                _,_,x = model.model(img)
                features = F.normalize(x)
                logits = header(features)

                loss_val = criterion(logits, label.float())

                loss_val.backward()
                
                clip_grad_norm_(model.model.parameters(), max_norm=5)
                clip_grad_norm_(header.parameters(), max_norm=5)

                opt_backbone.step()
                opt_header.step()
                model.set_zero()
                opt_backbone.zero_grad()
                opt_header.zero_grad()
                loss_meter.update(loss_val.item(), 1)

                pbar.set_postfix({
                    "Loss": f"{loss_meter.avg:.4f}",
                })

            scheduler_backbone.step(epoch)
            scheduler_header.step(epoch)

            logger.info(f"Epoch {epoch+1} finished. Avg Loss: {loss_meter.avg:.4f}")
            logger.info(f"Learning Rate after epoch {epoch+1}: {opt_header.param_groups[0]['lr']:.6f}")

            # Validate
            validate(model.model, header, val_loader, device, logger)

            # Save checkpoint
            callback_checkpoint(global_step, model.model, header)
            logger.info(f"Checkpoint saved at step {global_step}")
        validate(model.model, header, test_loader, device, logger)



def test(backbone, header, test_loader, device, logger):
    backbone.to(device)
    header.to(device)
    backbone.eval()
    header.eval()

    all_preds = []
    all_targets = []

    total_correct = 0
    total_labels = 0

    # Initialize correct predictions per class
    per_class_correct = None
    total_samples = 0

    for imgs, labels in tqdm(test_loader, desc="Testing", ncols=50):
        imgs = imgs.to(device)
        labels = labels.to(device)

        _, _, x = backbone(imgs)
        features = F.normalize(x)
        logits = header(features)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        if per_class_correct is None:
            num_classes = labels.size(1)
            per_class_correct = torch.zeros(num_classes, dtype=torch.long)

        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu().int())

        # Total correct overall
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_labels += labels.numel()

        # Per-class correct (matches regardless of class being 0 or 1)
        per_class_correct += (preds == labels).sum(dim=0).cpu()

        total_samples += labels.size(0)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Calculate metrics
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    mean_accuracy = total_correct / total_labels

    # Compute per-class accuracy (correct predictions / total samples)
    per_class_accuracy = per_class_correct.float() / total_samples

    # Calculate per-class confusion matrices
    classes = celeba_classes()
    confusion_matrices = {}
    
    for i, class_name in enumerate(classes):
        # For each class, calculate binary confusion matrix
        cm = confusion_matrix(all_targets[:, i], all_preds[:, i])
        confusion_matrices[class_name] = cm
        
        # Print class-specific metrics
        tn, fp, fn, tp = cm.ravel()
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        class_roc=roc_curve(all_targets[:, i], all_preds[:, i])
        logger.info(f"\nClass: {class_name}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Precision: {class_precision:.4f}")
        logger.info(f"Recall: {class_recall:.4f}")
        logger.info(f"F1: {class_f1:.4f}")

    # Print overall metrics
    logger.info(f"\nOverall Metrics:")
    logger.info(f"Mean Accuracy: {mean_accuracy:.4f}")
    logger.info(f"Macro Precision: {precision:.4f}")
    logger.info(f"Macro Recall: {recall:.4f}")
    logger.info(f"Macro F1 Score: {f1:.4f}")
    
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"  Class {classes[i]}: {acc:.4f}")
    return per_class_accuracy, confusion_matrices, all_targets, all_preds, mean_accuracy, precision, recall, f1



def evaluate(cfg, seed, sparsity):
    device = torch.device("cuda:0")

    weight_path= cfg.output + '_' + str(seed) + '_' + str(sparsity)
    output=os.path.join('output', 'swin_celeba_' + str(seed) + '_' + str(sparsity))
    
    backbone_path = os.path.join(weight_path, '12710backbone.pth')
    header_path = os.path.join(weight_path, '12710header.pth')
    os.makedirs(output, exist_ok=True)
    logger = setup_logger(output, training=False)


    _, _, test_set = full_celeba_get_datasets(
        cfg.data_path,
        return_test=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SWIN_prune(cfg, None, None)
    model.compress('slimming', sparsity)

    header = losses.MultiLabelHeader(
        in_features=cfg.embedding_size,
        out_features=40
    )

    model.model.load_state_dict(torch.load(backbone_path))
    header.load_state_dict(torch.load(header_path))

    test(
        model.model,
        header,
        test_loader,
        device,
        logger
    )




def main():
    parser = argparse.ArgumentParser(description="CelebA SWIN pruning")

    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Pruning sparsity values"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 7, 14, 16, 33, 42, 44, 55, 63, 101],
        help="Random seeds"
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation only"
    )

    args = parser.parse_args()

    for seed in args.seeds:
        for sparsity in args.sparsity:

            print(f"Seed: {seed}")
            print(f"Sparsity: {sparsity}")

            if args.eval:
                evaluate(cfg, seed, sparsity)
            else:
                train(cfg, seed, sparsity)


if __name__ == "__main__":
    main()
