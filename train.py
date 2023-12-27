from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import PIL
import os
import json
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter


batch_size = 8
n_classes = 21
num_epochs = 100
save_steps = 5
validation_batches = 10
save_folder = os.path.join(os.path.expanduser('~'), "data/christian/model_output")

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_images_to_tensorboard(model, loader, device, writer, epoch, n_images=10):
    model.eval()
    images, predictions = [], []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= n_images:  # Only infer on n_images
                break
            inputs = inputs.to(device)
            outputs = model(inputs)['out']
            pred = torch.argmax(outputs, dim=1).cpu()  # [N, H, W]

            images.append(inputs.cpu())
            predictions.append(pred)

    # # Convert list of tensors to single tensor
    # images_tensor = torch.stack(images)  # [N, C, H, W]
    # preds_tensor = torch.stack(predictions)  # [N, H, W]

    # # Add channel dimension to predictions and convert to float
    # preds_tensor = preds_tensor.unsqueeze(1).float()  # [N, 1, H, W]
    # preds_tensor /= preds_tensor.max()  # Normalize to range [0, 1]

    # # Convert to grid
    # images_grid = torchvision.utils.make_grid(images_tensor)
    # preds_grid = torchvision.utils.make_grid(preds_tensor)

    # # Add images to TensorBoard
    # writer.add_image('Validation/Images', images_grid, epoch)
    # writer.add_image('Validation/Predictions', preds_grid, epoch)

def save_checkpoint(model, optimizer, epoch, file_path):
    """
    Save model checkpoint.

    :param model: The PyTorch model to save.
    :param optimizer: The optimizer used during training.
    :param epoch: Current epoch number.
    :param file_path: Path to save the checkpoint.
    """
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(state, file_path)
    print(f"Checkpoint saved at epoch {epoch}")

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, split, transform=None, use_normals=True):
        if use_normals:
            self.inputs_dir = os.path.join(dataset_dir, 'normals')
        else:
            self.inputs_dir = os.path.join(dataset_dir, 'heights')
        self.labels_dir = os.path.join(dataset_dir, 'labels')
        self.use_normals = use_normals
        self.transform = transform
        self.images = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        if self.use_normals:
            input_img_name = img_name + "_bev_normals.png" 
        else:
            input_img_name = img_name + "_bev_height.png"
        label_img_name = img_name + "_bev_output_labels.png"
        
        img_path = os.path.join(self.inputs_dir, input_img_name)
        label_path = os.path.join(self.labels_dir, label_img_name)

        image = Image.open(img_path).convert("RGB").resize((401, 401), PIL.Image.NEAREST)
        label = Image.open(label_path).convert("L").resize((401, 401), PIL.Image.NEAREST)

        if self.transform:
            
            # im_arr = np.array(image)
            # im_arr32 = im_arr.astype(np.float32)/255
            # image = self.transform(im_arr32)
            image = self.transform(image)
            label = self.transform(label)
            
        label = label.squeeze(0).long()

        return image, label

def main():
    # Load pretrained DeepLabV3 model
    model = deeplabv3_resnet50(pretrained=True, weights="DeepLabV3_ResNet50_Weights.DEFAULT")
    # If your number of classes is 'n_classes'
    model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.to(device)  # Move model to CUDA device

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add other transformations as needed
    ])
    if os.listdir(save_folder):
        output_folder = str(max([int(f) for f in os.listdir(save_folder)]) + 1)
    else:
        output_folder = "0"
    save_path = os.path.join(save_folder, output_folder)
    os.makedirs(save_path)
    
    # Initialize TensorBoard Summary Writer
    writer = SummaryWriter(save_path)  # Adjust path as needed
    # Load the train, validation, test split
    split = json.load(open(os.path.join(os.path.expanduser('~'), "data/christian/bev/dataset_splits.json")))
    # Train
    # Create the dataset
    train_dataset = CustomDataset(dataset_dir=os.path.join(os.path.expanduser('~'), "data/christian/bev"), transform=transform, split=split["train"], use_normals=False)
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Validation
    # Create the dataset
    validation_dataset = CustomDataset(dataset_dir=os.path.join(os.path.expanduser('~'), "data/christian/bev"), transform=transform, split=split["validation"])
    # DataLoader
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # Training
        for inputs, labels in tqdm(train_loader, desc="training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}')
        
        # Log training loss
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(validation_loader, desc="training")):
                if i >= validation_batches:  # Limit the number of batches for validation
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)['out']
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

        avg_validation_loss = validation_loss / validation_batches
        print(f'Epoch {epoch+1}, Validation Loss: {avg_validation_loss}')
        
        # Log validation loss
        writer.add_scalar('Loss/validation', avg_validation_loss, epoch)
        
        # Perform inference and add images to TensorBoard
        add_images_to_tensorboard(model, validation_loader, device, writer, epoch)
        
        # Save checkpoint
        if (epoch + 1) % save_steps == 0 or (epoch + 1) == 1:
            checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
            
    writer.close()

        
if __name__ == "__main__":
    main()
