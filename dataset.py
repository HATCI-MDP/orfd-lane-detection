"""
ORFD Dataset loader for off-road lane detection
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class ORFDDataset(Dataset):
    """Off-Road Freespace Detection Dataset"""

    def __init__(self, root_dir, split='training', img_size=(256, 256), augment=False):
        """
        Args:
            root_dir: Path to ORFD dataset root
            split: 'training', 'validation', or 'testing'
            img_size: Target image size (height, width)
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.image_dir = os.path.join(root_dir, split, 'image_data')
        self.mask_dir = os.path.join(root_dir, split, 'gt_image')

        # Get list of images
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])

        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # For inference (no normalization)
        self.img_transform_raw = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load mask
        mask_name = img_name.replace('.png', '_fillcolor.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')

        # Data augmentation
        if self.augment:
            image, mask = self._augment(image, mask)

        # Resize mask
        mask = mask.resize(self.img_size, Image.NEAREST)

        # Transform image
        image = self.img_transform(image)

        # Convert mask to tensor (only 255 = road, ignore 128 which is sky)
        mask_arr = np.array(mask)
        mask_arr = (mask_arr == 255).astype(np.float32)  # Only 255 is road
        mask = torch.from_numpy(mask_arr).float().unsqueeze(0)

        return image, mask

    def _augment(self, image, mask):
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random brightness/contrast (image only)
        if np.random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(np.random.uniform(0.8, 1.2))

        if np.random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(np.random.uniform(0.8, 1.2))

        return image, mask

    def get_original_image(self, idx):
        """Get original image without normalization for visualization"""
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        return image, img_name


def get_dataloaders(data_root, batch_size=8, img_size=(256, 256), num_workers=4):
    """Create train, validation, and test dataloaders"""

    train_dataset = ORFDDataset(data_root, split='training', img_size=img_size, augment=True)
    val_dataset = ORFDDataset(data_root, split='validation', img_size=img_size, augment=False)
    test_dataset = ORFDDataset(data_root, split='testing', img_size=img_size, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    data_root = 'datasets/ORFD'
    dataset = ORFDDataset(data_root, split='training')
    print(f"Dataset size: {len(dataset)}")

    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
