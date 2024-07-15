# Data Augmentation and Cross-Validation using AlexNet

This project performs data augmentation and 5-fold cross-validation on a dataset using the AlexNet model for binary classification. The code includes custom data augmentation techniques, a custom dataset class, and training routines for the AlexNet model with PyTorch.

## Table of Contents

- [Data Augmentation and Cross-Validation using AlexNet](#data-augmentation-and-cross-validation-using-alexnet)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Data Preparation](#data-preparation)
  - [Data Augmentation](#data-augmentation)
  - [Custom Dataset Class](#custom-dataset-class)
  - [Training and Cross-Validation](#training-and-cross-validation)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Usage](#usage)
  - [Acknowledgements](#acknowledgements)
  - [Contributions](#contributions)
  - [Contacts](#contacts)


## Requirements

- Python 3.7+
- Libraries:
  - numpy
  - scipy
  - matplotlib
  - Pillow
  - requests
  - torch
  - torchvision
  - scikit-learn

Install the required libraries using pip:

pip install numpy scipy matplotlib Pillow requests torch torchvision scikit-learn


## Data Preparation

1. Download the dataset file from the provided URL and save it locally.
2. Load the dataset using scipy.io.loadmat.
3. Extract the training and test patterns along with their labels.
4. Ensure labels are in the range [0, 1].

url = 'https://www.dropbox.com/s/elfn1jd63k94mlr/DatasColor_29.mat?dl=1'
response = requests.get(url)
local_filename = 'DatasColor_29.mat'

# Save the file locally
with open(local_filename, 'wb') as f:
    f.write(response.content)

# Load data
mat_data = scipy.io.loadmat(local_filename)
data = mat_data['DATA']

## Data Augmentation

Various data augmentation techniques are applied to the dataset, including:

- Horizontal Flip
- Random Rotation
- Random Crop
- Shifting
- Color Jittering
- Adding Noise
- PCA Jittering

```python
def augment_data(images, labels):
    # Data augmentation techniques
    pass

def pca_jitter(img):
    # PCA jittering technique
    pass
```

## Custom Dataset Class

A custom dataset class CustomDataset is created to handle the augmented images and labels.

```python
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
```

## Training and Cross-Validation

The training and cross-validation process is performed using the AlexNet model. The training routine includes:

- Defining data loaders for training and validation sets.
- Initializing the model, criterion, and optimizer.
- Training and validation loops for each epoch.

```python
# Define the number of folds
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

# Training parameters
num_epochs = 10
for fold, (train_ids, val_ids) in enumerate(kfold.split(augmented_images)):
    # Sample elements randomly for this fold
    # Define data loaders for training and validation
    # Initialize the model, criterion, and optimizer for each fold
    # Training and validation loops
    pass
```

## Evaluation

After training, the model is evaluated on the augmented training set to ensure accuracy.

```python
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
```

## Results

The average training and validation loss and accuracy are calculated and plotted for each fold. The results provide insights into the model's performance across different folds.

```python
# Plotting the training and validation losses and accuracies for each fold
plt.figure(figsize=(14, 6))

# Plot losses
plt.subplot(1, 2, 1)
for fold in range(k_folds):
    plt.plot(train_losses[fold], label=f'Fold {fold+1} Train Loss')
    plt.plot(val_losses[fold], label=f'Fold {fold+1} Val Loss', linestyle='--')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
for fold in range(k_folds):
    plt.plot(train_accuracies[fold], label=f'Fold {fold+1} Train Accuracy')
    plt.plot(val_accuracies[fold], label=f'Fold {fold+1} Val Accuracy', linestyle='--')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```

## Usage

Ensure all required libraries are installed.
Run the script to download the dataset, augment the data, and perform training and cross-validation.
Observe the training and validation loss and accuracy plots to understand the model's performance.

```bash
python script_name.py
```

## Acknowledgements

This code is based on the AlexNet model and uses data augmentation techniques to enhance the dataset. The project demonstrates the use of cross-validation to evaluate model performance effectively.

## Contributions
Feel free to contribute to this project by submitting pull requests or reporting issues.

## Contacts
For any questions, please contact [jacopo.righetto@studenti.unipd.it], [giacomo.sanguin@studenti.unipd.it].