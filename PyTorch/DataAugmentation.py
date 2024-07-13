import random
from PIL import Image, ImageOps, ImageEnhance
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import KFold

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data augmentation function
def augment_data(images, labels):
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        # Technique 1: Original Image
        augmented_images.append(img)
        augmented_labels.append(label)

        # Technique 2: Horizontal Flip
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(flipped_img)
        augmented_labels.append(label)

        # Technique 3: Random Rotation
        rotated_img = img.rotate(random.randint(-30, 30), resample=Image.BILINEAR)
        augmented_images.append(rotated_img)
        augmented_labels.append(label)

        # Technique 4: Random Crop
        crop_size = (random.randint(0, img.size[0] - 1), random.randint(0, img.size[1] - 1))
        cropped_img = img.crop((0, 0, crop_size[0], crop_size[1])).resize(img.size)
        augmented_images.append(cropped_img)
        augmented_labels.append(label)

        # Technique 5: Shifting
        shifted_img = img.transform(img.size, Image.AFFINE, (1, 0, random.randint(-10, 10), 0, 1, random.randint(-10, 10)))
        augmented_images.append(shifted_img)
        augmented_labels.append(label)

        # Technique 6: Color Jittering
        jittered_img = ImageEnhance.Contrast(img).enhance(1 + (random.random() - 0.5) * 0.4)
        jittered_img = ImageEnhance.Color(jittered_img).enhance(1 + (random.random() - 0.5) * 0.4)
        jittered_img = ImageEnhance.Brightness(jittered_img).enhance(1 + (random.random() - 0.5) * 0.4)
        augmented_images.append(jittered_img)
        augmented_labels.append(label)

        # Technique 7: Adding Noise
        img_array = np.array(img)
        noisy_img = img_array + np.random.normal(0, 0.01 * 255, img_array.shape)
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img)
        augmented_images.append(noisy_img)
        augmented_labels.append(label)

        # Technique 8: PCA Jittering
        pca_jittered_img = pca_jitter(img)
        augmented_images.append(pca_jittered_img)
        augmented_labels.append(label)

        # Technique 9: Random Rotation and Adding Noise
        rotated_img = img.rotate(random.randint(-30, 30), resample=Image.BILINEAR)
        rotated_img_array = np.array(rotated_img)
        noisy_rotated_img = rotated_img_array + np.random.normal(0, 0.01 * 255, rotated_img_array.shape)
        noisy_rotated_img = np.clip(noisy_rotated_img, 0, 255).astype(np.uint8)
        noisy_rotated_img = Image.fromarray(noisy_rotated_img)
        augmented_images.append(noisy_rotated_img)
        augmented_labels.append(label)

        # Technique 10: Horizontal Flip and PCA Jittering
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        pca_jittered_flipped_img = pca_jitter(flipped_img)
        augmented_images.append(pca_jittered_flipped_img)
        augmented_labels.append(label)

        # Technique 11: Random Crop, Shift, and Add Noise
        crop_size = (random.randint(0, img.size[0] - 1), random.randint(0, img.size[1] - 1))
        cropped_img = img.crop((0, 0, crop_size[0], crop_size[1])).resize(img.size)
        shifted_img = cropped_img.transform(img.size, Image.AFFINE, (1, 0, random.randint(-10, 10), 0, 1, random.randint(-10, 10)))
        shifted_img_array = np.array(shifted_img)
        noisy_shifted_img = shifted_img_array + np.random.normal(0, 0.01 * 255, shifted_img_array.shape)
        noisy_shifted_img = np.clip(noisy_shifted_img, 0, 255).astype(np.uint8)
        noisy_shifted_img = Image.fromarray(noisy_shifted_img)
        augmented_images.append(noisy_shifted_img)
        augmented_labels.append(label)

        # Technique 12: Horizontal Flip, Random Rotation, and Color Jittering
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        rotated_img = flipped_img.rotate(random.randint(-30, 30), resample=Image.BILINEAR)
        jittered_img = ImageEnhance.Contrast(rotated_img).enhance(1 + (random.random() - 0.5) * 0.4)
        jittered_img = ImageEnhance.Color(jittered_img).enhance(1 + (random.random() - 0.5) * 0.4)
        jittered_img = ImageEnhance.Brightness(jittered_img).enhance(1 + (random.random() - 0.5) * 0.4)
        augmented_images.append(jittered_img)
        augmented_labels.append(label)

    return augmented_images, augmented_labels

def pca_jitter(img):
    img = np.array(img, dtype=np.float32)
    img_flat = img.reshape(-1, 3)
    mean = np.mean(img_flat, axis=0)
    img_flat -= mean
    cov = np.cov(img_flat, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    jitter = np.random.randn(3) * 0.1
    jittered_img_flat = img_flat + jitter @ eigvecs.T
    jittered_img_flat += mean
    jittered_img = jittered_img_flat.reshape(img.shape)
    jittered_img = np.clip(jittered_img, 0, 255).astype(np.uint8)
    return Image.fromarray(jittered_img)

# Load data
mat_data = scipy.io.loadmat('/Users/giacomosanguin/Documents/GitHub/DeepLearning/PyTorch/DatasColor_29.mat')
data = mat_data['DATA']

NF = data[0, 2].shape[0]  # number of folds
DIV = data[0, 2]  # division of training and test sets
DIM1 = data[0, 3][0, 0]  # number of training patterns
DIM2 = data[0, 4][0, 0]  # number of patterns
yE = data[0, 1][0]  # labels
NX = data[0, 0][0]  # images

# Prepare the data
train_patterns = DIV[0, :DIM1].flatten() - 1
test_patterns = DIV[0, DIM1:DIM2].flatten() - 1
train_labels = yE[train_patterns].astype(int) - 1
test_labels = yE[test_patterns].astype(int) - 1

# Ensure labels are in range [0, 1]
train_labels = np.where(train_labels > 1, 1, train_labels)
test_labels = np.where(test_labels > 1, 1, test_labels)

train_images = [Image.fromarray(NX[i].astype(np.uint8)) for i in train_patterns]
test_images = [Image.fromarray(NX[i].astype(np.uint8)) for i in test_patterns]

# Augment data
augmented_images, augmented_labels = augment_data(train_images, train_labels)

# Create PyTorch dataset
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])
train_dataset = CustomDataset(augmented_images, augmented_labels, transform=transform)

# Define the number of folds
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

# Initialize lists to hold training and validation losses for each fold
train_losses = []
val_losses = []

# Training parameters
num_epochs = 10
for fold, (train_ids, val_ids) in enumerate(kfold.split(augmented_images)):
    print(f'Fold {fold+1}/{k_folds}')
    
    # Sample elements randomly for this fold
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    
    # Define data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=30, sampler=train_subsampler)
    val_loader = DataLoader(train_dataset, batch_size=30, sampler=val_subsampler)
    
    # Initialize the model, criterion, and optimizer for each fold
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # Save training and validation losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    print(f'Finished Fold {fold+1}/{k_folds}')

# Calculate average loss across folds
avg_train_loss = sum(train_losses) / len(train_losses)
avg_val_loss = sum(val_losses) / len(val_losses)
print(f'Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}')

# Evaluation (just for the purpose of completeness, actual testing would be similar)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in DataLoader(train_dataset, batch_size=30, shuffle=True):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy on augmented training set: {100 * correct / total:.2f}%')
