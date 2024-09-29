import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pdb

# Custom Dataset for loading images and masks
class KidneyDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths  # Can be None for unlabeled images
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mask_paths:
            mask = cv2.imread(self.mask_paths[idx], 0)  # Load mask as grayscale
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = np.where(mask == 255, 1, mask).astype(np.float32)  # left kidney
            mask = np.where(mask == 127, 2, mask).astype(np.float32)  # right kidney
            mask = np.clip(mask, 0, 2)  # Ensure values are within the range [0, 2]
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)  # Placeholder for unlabeled data

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# Define transformations for data augmentation
# def get_transforms():
#     return transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

# Load labeled and unlabeled image paths
def load_data(labeled_dir, unlabeled_dir):
    labeled_images = sorted(glob(os.path.join(labeled_dir, "images", "*.jpg")))
    labeled_masks = sorted(glob(os.path.join(labeled_dir, "masks", "*.png")))
    
    unlabeled_images = sorted(glob(os.path.join(unlabeled_dir, "images", "*.jpg")))

    return labeled_images, labeled_masks, unlabeled_images

# Define the weakly supervised deep learning model for segmentation
class WeaklySupervisedKidneySegmentation:
    def __init__(self, labeled_loader, unlabeled_loader, device):
        self.device = device
        self.model = deeplabv3_resnet50(pretrained=True)  # Pretrained model
        self.model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)  # Output for 3 classes: background, left kidney, right kidney
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

    def train_step(self, labeled_data):
        self.model.train()
        images, masks = labeled_data
        images, masks = images.to(self.device), masks.to(self.device)

        self.optimizer.zero_grad()
        #pdb.set_trace()
        outputs = self.model(images)['out']
        # outputs = outputs_['out']
        loss = self.criterion(outputs, masks.long()) # masks.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def unsupervised_loss(self, unlabeled_data, model, threshold=0.5, consistency_weight=1.0):
        """
        Calculate the unsupervised loss for instance segmentation using weakly supervised techniques.
        
        Args:
            unlabeled_data (torch.Tensor): Batch of unlabeled images.
            model (torch.nn.Module): The segmentation model.
            threshold (float): Threshold for pseudo-labeling.
            consistency_weight (float): Weight for the consistency loss.
        
        Returns:
            torch.Tensor: The unsupervised loss.
        """
        # Generate pseudo-labels using the model
        with torch.no_grad():
            # pdb.set_trace()
            pseudo_labels = model(unlabeled_data)['out']
            # pseudo_labels = (pseudo_labels > threshold).float()
            pseudo_labels = torch.argmax(pseudo_labels, dim=1).float()
        
        # Apply augmentations to the unlabeled data
        augmented_data = self.apply_augmentations(unlabeled_data)
        
        # Get model predictions for the augmented data
        augmented_predictions = model(augmented_data)['out']
        augmented_predictions = torch.argmax(augmented_predictions, dim=1).float()

        # Ensure the shapes match
        if augmented_predictions.shape != pseudo_labels.shape:
            raise ValueError(f"Shape mismatch: augmented_predictions shape {augmented_predictions.shape} does not match pseudo_labels shape {pseudo_labels.shape}")
    
        
        # Calculate the consistency loss
        consistency_loss = F.mse_loss(augmented_predictions, pseudo_labels)
        
        # Combine the losses
        total_loss = consistency_weight * consistency_loss
        
        return total_loss

    def apply_augmentations(self, images):
        """
        Apply augmentations to the input images.
        
        Args:
            images (torch.Tensor): Batch of images.
        
        Returns:
            torch.Tensor: Batch of augmented images.
        """
        # Example augmentation: horizontal flip
        augmented_images = torch.flip(images, dims=[3])
        
        return augmented_images

    def train(self, epochs):
        best_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(epochs):
            labeled_loss_total = 0.0
            for labeled_batch in tqdm(self.labeled_loader):
                labeled_loss_total += self.train_step(labeled_batch)

            # Optionally add weakly supervised loss with unlabeled data
            for unlabeled_batch in tqdm(self.unlabeled_loader):
                unlabeled_images, _ = unlabeled_batch  # Extract images from the batch
                unlabeled_images = unlabeled_images.to(self.device)
                # Debug statement to check input dimensions
                print(f"Unlabeled images shape: {unlabeled_images.shape}")
                # Compute and update based on weak supervision strategy
                self.unsupervised_loss(unlabeled_images, model=self.model)

            print(f"Epoch [{epoch+1}/{epochs}], Labeled Loss: {labeled_loss_total:.4f}")

            # Check for improvement
            if labeled_loss_total < best_loss:
                best_loss = labeled_loss_total
                epochs_since_improvement = 0
                # Save the best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("Saved best model")
            else:
                epochs_since_improvement += 1

            # Save the latest model
            torch.save(self.model.state_dict(), 'latest_model.pth')
            print("Saved latest model")

            # Early stopping
            if epochs_since_improvement >= 5:
                print("Early stopping triggered")
                break

# Main function
if __name__ == "__main__":
    # Directories containing images
    labeled_dir = "./data/labeled"
    unlabeled_dir = "./data/unlabeled"

    # Load data
    labeled_images, labeled_masks, unlabeled_images = load_data(labeled_dir, unlabeled_dir)

    # DataLoader for labeled and unlabeled data
    labeled_dataset = KidneyDataset(labeled_images, labeled_masks, transform=get_transforms())
    unlabeled_dataset = KidneyDataset(unlabeled_images, transform=get_transforms())

    # Debug statements to check dataset lengths
    print(f"Number of labeled samples: {len(labeled_dataset)}")
    print(f"Number of unlabeled samples: {len(unlabeled_dataset)}")
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=4, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the weakly supervised segmentation model
    model = WeaklySupervisedKidneySegmentation(labeled_loader, unlabeled_loader, device)
    model.train(epochs=50)
