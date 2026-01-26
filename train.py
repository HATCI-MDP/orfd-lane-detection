"""
Training script for U-Net based off-road lane detection
Faster and lighter alternative to SegFormer
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import ORFDDataset
from model import UNet, UNetSmall, CombinedLoss, calculate_iou, calculate_dice


def get_dataloaders(data_root, batch_size=8, img_size=256, num_workers=0):
    """Create train, validation, and test dataloaders"""
    train_dataset = ORFDDataset(data_root, split='training', img_size=(img_size, img_size), augment=True)
    val_dataset = ORFDDataset(data_root, split='validation', img_size=(img_size, img_size), augment=False)
    test_dataset = ORFDDataset(data_root, split='testing', img_size=(img_size, img_size), augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks)
        total_dice += calculate_dice(outputs, masks)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{calculate_iou(outputs, masks):.4f}'
        })

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
            total_dice += calculate_dice(outputs, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def save_checkpoint(model, optimizer, epoch, loss, iou, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'iou': iou,
    }, path)


def plot_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(history['train_iou'], label='Train')
    axes[1].plot(history['val_iou'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('IoU')
    axes[1].legend()

    axes[2].plot(history['train_dice'], label='Train')
    axes[2].plot(history['val_dice'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].set_title('Dice Score')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'unet_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    # Model
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1)
        print("\nUsing U-Net (31M params)")
    else:
        model = UNetSmall(in_channels=3, out_channels=1)
        print("\nUsing U-Net Small (7.7M params)")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Load pretrained weights if provided
    if args.pretrained:
        print(f"Loading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and args.amp else None

    # Training history
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    }

    best_val_iou = 0
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_iou)

        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_iou,
                os.path.join(output_dir, 'best_model.pth')
            )
            print(f"  -> New best model saved! IoU: {val_iou:.4f}")
        else:
            patience_counter += 1

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_iou,
            os.path.join(output_dir, 'latest_model.pth')
        )

        # Plot history
        plot_history(history, os.path.join(output_dir, 'training_history.png'))

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {patience_counter} epochs without improvement")
            break

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final evaluation on test set...")

    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_iou, test_dice = validate(model, test_loader, criterion, device)
    print(f"Test - Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, Dice: {test_dice:.4f}")

    print(f"\nTraining complete! Best validation IoU: {best_val_iou:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net for off-road lane detection')

    parser.add_argument('--data_root', type=str, default='datasets/ORFD',
                        help='Path to ORFD dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for models')
    parser.add_argument('--model', type=str, default='unet_small',
                        choices=['unet', 'unet_small'],
                        help='Model: unet (31M) or unet_small (7.7M)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained U-Net checkpoint')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size (default 256)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default 8)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Data loading workers')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')

    args = parser.parse_args()
    main(args)
