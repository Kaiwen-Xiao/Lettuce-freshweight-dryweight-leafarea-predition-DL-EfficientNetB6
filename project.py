import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import cv2
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import os
import multiprocessing
from math import sqrt

# number of system CPU
cpu_count = os.cpu_count()

# number of physical CPU
physical_cpu_count = multiprocessing.cpu_count()

# number of GPU
gpu_count = torch.cuda.device_count()
print("CPU cores:", cpu_count)
print("Physical CPU cores:", physical_cpu_count)
print("Available GPUs:", gpu_count)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
torch.manual_seed(0)
np.random.seed(67403538)

# Define the transforms for RGB images
rgb_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((1080, 1080)),
    transforms.Resize((528, 528)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flipping horizontally
    transforms.RandomVerticalFlip(p=0.5),    # 50% chance of flipping vertically
    transforms.RandomRotation(degrees=15),   # Rotate by up to ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

validation_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((1080, 1080)),
    transforms.Resize((528, 528)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset class
class LettuceDataset(Dataset):
    def __init__(self, json_file, rgb_dir, transform=validation_transform):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            rgb_dir (string): Directory with all the RGB images.
            depth_dir (string): Directory with all the Depth images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.rgb_dir = rgb_dir
        self.transform = transform
        # Load data from json
        with open(json_file) as f:
            self.annotations = json.load(f)["Measurements"]  # Adjust based on actual structure of JSON
            
    def __len__(self):
        return len(self.annotations)        
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        # Construct the paths for the RGB and Depth images
        rgb_image_name = annotation['RGB_Image']
        rgb_image_path = f'{self.rgb_dir}/{rgb_image_name}'
        # Load the RGB image
        rgb_image = cv2.imread(rgb_image_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Center crop and resize both RGB and depth images
        center_x, center_y = rgb_image.shape[1] // 2, rgb_image.shape[0] // 2
        half_size = 540  # half of the new size 1080 / 2
        cropped_rgb_image = rgb_image[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]
        # cropped_depth_image = depth_image[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

        # Preprocess and stack depth with RGB
        if self.transform:
          cropped_rgb_image = self.transform(cropped_rgb_image)
        # Get labels
        labels = torch.tensor([annotation['FreshWeightShoot'], annotation['DryWeightShoot'], annotation['LeafArea']])
        # return stacked_image, labels
        return cropped_rgb_image, labels

# Training function with early stopping and checkpointing
def train_model(model, criterion, optimizer, train_loader, val_loader, scheduler,epochs=25, device=device):
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # model.eval()
        # val_running_loss = 0.0
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         inputs = inputs.to(device)
        #         labels = labels.to(device).float()
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         val_running_loss += loss.item() * inputs.size(0)
        
        # val_epoch_loss = val_running_loss / len(val_loader.dataset)
        # log_text = f'Epoch {epoch}/{epochs - 1}: Loss: {epoch_loss:.4f} - Val Loss: {val_epoch_loss:.4f}'
        # print(log_text)
        # with open("log.txt", "a") as file:
        #     file.write(log_text+ "\n")
        # scheduler.step(val_epoch_loss)
        model.eval()
        val_running_loss = 0.0
        actuals = []  # actual labels
        predictions = []  # predictions

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                actuals.extend(labels.cpu().numpy())
                predictions.extend(outputs.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)

        # calculate RMSE and R^2
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        rmse = sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        log_text = f'Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f} - Val Loss: {val_epoch_loss:.4f} - Val RMSE: {rmse:.4f} - Val R^2: {r2:.4f}'
        print(log_text)

        # with open("log-b6.txt", "a") as file:
        #     file.write(log_text+ "\n")
        
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.state_dict(), 'best_model_weights-b6.pth')
            patience_counter = 0
            print('Validation loss decreased, saving model...')
            with open("log-b6.txt", "a") as file:
                file.write("\n" + "Validation loss decreased"+ "\n")
        else:
            patience_counter += 1
            print(f'Validation loss did not improve. Patience counter {patience_counter}/{patience}')
            with open("log-b6.txt", "a") as file:
                file.write("\n" + "Validation loss did not improve"+ "\n")
        with open("log-b6.txt", "a") as file:
            file.write(log_text+ "\n")
        # if patience_counter >= patience:
        #     print('Early stopping triggered.')
        #     break
    
    return model




      
def main():           
  # Initialize the dataset
  json_file = 'GroundTruth/GroundTruth_All_388_Images.json'  # Replace with your json file path
  rgb_dir = 'RGBImages'   # Replace with your RGB images directory
  # depth_dir = 'DepthImages'  # Replace with your Depth images directory

  # dataset = LettuceDataset(json_file=json_file, rgb_dir=rgb_dir, transform=rgb_transform)
    
  with open(json_file) as f:
    annotations = json.load(f)["Measurements"]
  annotations_list = list(annotations.values())
  train_annotations, val_annotations = train_test_split(annotations_list, test_size=0.1, random_state=42)
  # print(val_annotations)
  train_set = LettuceDataset(json_file, rgb_dir, transform=rgb_transform)
  train_set.annotations = train_annotations
    
  val_set = LettuceDataset(json_file, rgb_dir,transform=validation_transform)
  val_set.annotations = val_annotations

  # Load data
  train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
  
  
  
  # Load the pre-trained EfficientNet model
  model = EfficientNet.from_pretrained('efficientnet-b6')
  # Adjust the final fully connected layer
  in_features = model._fc.in_features
  model._fc = nn.Linear(in_features, 3)
  try:
    model.load_state_dict(torch.load('best_model_weights-b6.pth', map_location=device))
    print("Loaded model weights from 'best_model_weights-b6.pth'.")
  except FileNotFoundError:
      print("No weights file found at 'best_model_weights-b6.pth', starting training from scratch.")
  model.to(device)
   
  #Loss 
  criterion = nn.MSELoss()
    
  # #Learning rate
  Learing_rate = 5.0e-12
  for i in range (10):
      print(f'learing_rate: {Learing_rate}')
      optimizer = optim.RMSprop(model.parameters(), Learing_rate)
      op_text = f'RMS mode Start'
      with open("log-b6.txt", "a") as file:
        file.write(op_text+ "\n")
      scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
      model = train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, epochs=25)   
        
      optimizer = optim.Adam(model.parameters(), Learing_rate)
      op_text = f'Adam mode Start'
      with open("log-b6.txt", "a") as file:
        file.write(op_text+ "\n")
      scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
      model = train_model(model, criterion, optimizer, train_loader, val_loader, scheduler, epochs=25)
      Learing_rate /= 10
  return True

if __name__ == "__main__":
  main()