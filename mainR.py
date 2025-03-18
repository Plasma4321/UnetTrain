# Importing the modules from the UNet folder
from UNet.unet.unet_model import UNet
from DeepLabV3Plus.metrics import StreamSegMetrics

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import logging

class LULCDataset(Dataset):
    def __init__(self, datadir, transform=None, gt_transform = None):
        self.datadir = datadir
        self.transform = transform
        self.gt_transform = gt_transform

        self.imdb = []
        for img in os.listdir(self.datadir+'/images'):
            image_name = img.split('.')[0]
            ext_name = img.split('.')[-1]
            img_path = os.path.join(self.datadir, 'images', img)
            gt_path = os.path.join(self.datadir, 'gts', image_name + '_gt.' + ext_name)
            self.imdb.append((img_path, gt_path))

    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):

        img_path, gt_path = self.imdb[idx]

        # Load images
        image = Image.open(img_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("L")  # Assuming GT is grayscale

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        label = np.array(gt_image)
        # print(np.unique(label))
  
        
        label = torch.LongTensor(label)  

        return image, label
    
       
def train(args, model, train_loader, test_loader, device, restart=False):
    model.to(device)
    start_epoch = 0

    best_iou_scores = [0.0, 0.0, 0.0]  # Best, 2nd best, 3rd best IoU
    best_checkpoints = [None, None, None]
    best_general_checkpoint = None
    best_general_iou = 0.0

    train_losses = []
    val_losses = []
    iou_scores = []


    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if restart:
        checkpoint = torch.load(args.checkpoint_path, map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # Load the best IoU from the best model checkpoint
        # Load best models if available
        for i, path in enumerate([args.best_model_path, args.best_model2_path, args.best_model3_path]):
            if os.path.exists(path):
                best_checkpoint = torch.load(path, map_location=torch.device('cpu'),weights_only=False)
                best_iou_scores[i] = best_checkpoint.get('best_iou', 0.0)
                best_checkpoints[i] = best_checkpoint
        else:
            best_iou_scores = [0.0, 0.0, 0.0]

        if os.path.exists(args.best_model_general_path):
            best_general_checkpoint = torch.load(args.best_model_general_path, map_location=torch.device('cpu'),weights_only=False)
            best_general_iou = best_general_checkpoint.get('best_iou', 0.0)
    else:
        # If not restarting, remove old loss and best model files
        for path in [args.best_model_path, args.best_model2_path, args.best_model3_path]:
            if os.path.exists(path):
                os.remove(path)

    

    for epoch in range(start_epoch, args.epochs):

        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch: {epoch+1} / {args.epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        

        # Calculate average train loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        metrics = StreamSegMetrics(args.num_classes)  # IoU calculator
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # Compute loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Compute IoU Score
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()  # Get predicted class
                targets = masks.cpu().numpy()
                metrics.update(targets, preds)

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        
        # Compute IoU metrics
        iou_results = metrics.get_results()
        avg_iou = iou_results["Mean IoU"]  # Extract mean IoU
        iou_scores.append(avg_iou)

        # Updating Schedular Step and logging the current learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Learning Rate: {current_lr:.6f}")

        # Save the last checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, args.checkpoint_path)

        # Store the model in best, second-best, or third-best if applicable
        if avg_iou > best_iou_scores[0]:
            best_iou_scores = [avg_iou, best_iou_scores[0], best_iou_scores[1]]
            best_checkpoints = [{
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': avg_iou,
                'best_train_loss': avg_train_loss,
                'best_val_loss': avg_val_loss
            }] + best_checkpoints[:2]
        elif avg_iou > best_iou_scores[1]:
            best_iou_scores[1:] = [avg_iou, best_iou_scores[1]]
            best_checkpoints[1:] = [{
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': avg_iou,
                'best_train_loss': avg_train_loss,
                'best_val_loss': avg_val_loss
            }] + best_checkpoints[1:2]
        elif avg_iou > best_iou_scores[2]:
            best_iou_scores[2] = avg_iou
            best_checkpoints[2] = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': avg_iou,
                'best_train_loss': avg_train_loss,
                'best_val_loss': avg_val_loss
            }

        # Save best model checkpoints
        for i, path in enumerate([args.best_model_path, args.best_model2_path, args.best_model3_path]):
            if best_checkpoints[i]:
                torch.save(best_checkpoints[i], path)

        # Save best model after 30 epochs
        if epoch >= 30 and avg_iou > best_general_iou:
            best_general_iou = avg_iou
            best_general_checkpoint = checkpoint
            torch.save(best_general_checkpoint, args.best_model_general_path)

    
def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225]) 
    ])

    gt_transform = transforms.Compose([    
        transforms.ToTensor()
    ])


    train_dir = os.path.join(args.datadir, 'train')
    test_dir = os.path.join(args.datadir, 'test')

    train_dataset = LULCDataset(train_dir, transform=transform, gt_transform = gt_transform)
    test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True, shuffle=True,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=2)

    model = UNet( n_channels=3, n_classes=args.num_classes, bilinear=False)

    model.to(device)

    if not args.eval_only:
        # train(args, model, train_loader, device, restart = args.restart)
        train(args, model, train_loader,test_loader, device, restart = args.restart)

if __name__ == '__main__':
    # Set up logging with a file handler
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('/home/temp/A100_Job/Checkpoints_Logs/LoggingData.log')  # Log to file 
        ]
    )

    logging.getLogger('PIL').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Unet training script")

#/home/temp/BingRGB/
#/home/temp/DataDummy

    parser.add_argument("--datadir", type=str, default="/home/temp/DataDummy", help="Path to the dataset directory")

    parser.add_argument("--batch_size", type=int, default=72, help="Batch size for training (default: 32)")

    parser.add_argument("--best_model_path", type=str, default="/home/temp/A100_Job/Checkpoints_Logs/best_model.pth",help="Path to save the best model checkpoint")#
    parser.add_argument("--best_model2_path", type=str, default="/home/temp/A100_Job/Checkpoints_Logs/best_model2.pth",help="Path to save the 2nd best model checkpoint")#
    parser.add_argument("--best_model3_path", type=str, default="/home/temp/A100_Job/Checkpoints_Logs/best_model3.pth",help="Path to save the 3rd best model checkpoint")#
    parser.add_argument("--best_model_general_path", type=str, default="/home/temp/A100_Job/Checkpoints_Logs/best_model_general.pth",help="Path to save the general best model checkpoint")#
    parser.add_argument("--checkpoint_path", type=str, default= '/home/temp/A100_Job/Checkpoints_Logs/checkpoint.pth', help='Saved model path')
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training (default: 0.001)")
    parser.add_argument("--num_classes", type = int, default=6, help="Number of classes")
    parser.add_argument("--eval_only", action='store_true', default=False, help = "Determining if only evaluation")
    parser.add_argument("--restart", default = False, action='store_true',  help="Determine if it should start from a checkpoint")

    args = parser.parse_args()
    main(args)

