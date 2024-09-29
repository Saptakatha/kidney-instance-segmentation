import os
import numpy as np
import cv2
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
from tqdm import tqdm
import argparse

# Define transformations for inference
def get_inference_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# Load the trained model
def load_model(model_path, device):
    model = deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=1) # Output for 3 classes: background, left kidney, right kidney

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove auxiliary classifier keys
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image_path, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image = augmented['image']
    return image.unsqueeze(0)  # Add batch dimension

# Perform inference
def infer(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)['out']
        # output = torch.sigmoid(output)
        # output = output.squeeze().cpu().numpy()
        # output = (output > 0.5).astype(np.uint8)  # Thresholding to get binary mask
        output = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Get the class with the highest score
    return output

# Main function for inference
def main_inference(image_paths, model_path, output_dir, device):
    transform = get_inference_transforms()
    model = load_model(model_path, device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path in tqdm(image_paths):
        image_tensor = preprocess_image(image_path, transform)
        mask = infer(model, image_tensor, device)
        
        # Save the output mask
        mask_path = os.path.join(output_dir, os.path.basename(image_path))
        mask = np.where(mask == 1, 255, mask).astype(np.uint8)  # Left kidney
        mask = np.where(mask == 2, 127, mask).astype(np.uint8)  # Right kidney
        cv2.imwrite(mask_path, mask)  # Save mask as image

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Inference for kidney segmentation")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Path to the test data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the predictions')
    args = parser.parse_args()

    # Paths to the unseen images
    unseen_image_paths = sorted(glob(os.path.join(args.test_data_dir, "*.jpg")))
    print(f"Number of unseen images: {len(unseen_image_paths)}")

    # Path to the trained model
    model_path = args.model_dir

    # Output directory to save the masks
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run inference
    main_inference(unseen_image_paths, model_path, output_dir, device)