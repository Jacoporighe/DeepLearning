import random
from PIL import Image
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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
        # Convert to PIL image if necessary
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        # Technique 1: Add Gaussian noise
        img_array = np.array(img)
        noisy_img = img_array + np.random.normal(0, 0.01, img_array.shape)
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img)
        augmented_images.append(noisy_img)
        augmented_labels.append(label)

        # Technique 2: Horizontal Flip and PCA Jittering
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        pca_jittered_img = pca_jitter(flipped_img)
        augmented_images.append(pca_jittered_img)
        augmented_labels.append(label)

        # Technique 3: Random Crop, Shift, and Add Noise
        crop_size = (random.randint(0, img.size[0] - 1), random.randint(0, img.size[1] - 1))
        cropped_img = img.crop((0, 0, crop_size[0], crop_size[1])).resize(img.size)
        shifted_img = cropped_img.transform(img.size, Image.AFFINE, (1, 0, random.randint(-10, 10), 0, 1, random.randint(-10, 10)))
        shifted_img = np.array(shifted_img)
        noisy_shifted_img = shifted_img + np.random.normal(0, 0.01, shifted_img.shape)
        noisy_shifted_img = np.clip(noisy_shifted_img, 0, 255).astype(np.uint8)
        noisy_shifted_img = Image.fromarray(noisy_shifted_img)
        augmented_images.append(noisy_shifted_img)
        augmented_labels.append(label)

        # Technique 4: Horizontal Flip, Random Rotation, and Color Jittering
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        rotated_img = flipped_img.rotate(random.randint(-30, 30))
        jittered_img = transforms.ColorJitter(contrast=0.2, saturation=0.2, brightness=0.2)(rotated_img)
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
mat_data = scipy.io.loadmat('DatasColor_29.mat')
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

# Create PyTorch dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])
train_dataset = CustomDataset(augmented_images, augmented_labels, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)

# Load pretrained network
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)  # Assuming binary classification

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataloader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Evaluation (mock evaluation for demonstration)
model.eval()
accuracy = 0.0
with torch.no_grad():
    for inputs, labels in train_dataloader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        accuracy += torch.sum(preds == labels.data)

accuracy = accuracy.double() / len(train_dataloader.dataset)
print(f'Accuracy: {accuracy:.4f}')
