# Computer Assignment 4: CNN Segmentation
## Alaqian Zafar - aaz7118

## Table of Contents
- <a href='#p1a'>Part (a)</a>
- <a href='#p1b'>Part (b)</a>
- <a href='#p1c'>Part (c)</a>
    - [Architecture](#Architecture)
    - [Loss Function](#Loss-Function)
- <a href='#p1d'>Part (d)</a>
- <a href='#p2a'>Part (e)</a>
- <a href='#p2b'>Part (f)</a>
- <a href='#p2c'>Part (g)</a>


```python
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

%matplotlib inline
```


```python
try:
    from google.colab import drive
    drive.mount('/content/drive')
    path = '/content/drive/MyDrive/ECE-GY 6123 Image and Video Processing/Computer Assignments/CA04/archive'
except:
    path = 'archive'
```

<a id='p1a'></a>
##### (a) Cut the FudanPed dataset into an 80-10-10 train-val-test split.

[Table of Contents](#Table-of-Contents)


```python
image_paths = sorted([os.path.join(path, "PNGImages", image) for image in os.listdir(os.path.join(path, "PNGImages"))])
mask_paths = sorted([os.path.join(path, "PedMasks", mask) for mask in os.listdir(os.path.join(path, "PedMasks"))])

indices = list(range(len(image_paths)))
train_indices = random.sample(indices, k=int(len(indices)*0.8))
val_indices = random.sample(set(indices)-set(train_indices), k=int(len(indices)*0.1))
test_indices = list(set(indices)-set(train_indices)-set(val_indices))

train_image_paths = [image_paths[i] for i in train_indices]
train_mask_paths = [mask_paths[i] for i in train_indices]
val_image_paths = [image_paths[i] for i in val_indices]
val_mask_paths = [mask_paths[i] for i in val_indices]
test_image_paths = [image_paths[i] for i in test_indices]
test_mask_paths = [mask_paths[i] for i in test_indices]

def square_pad(image):
    h_diff = max(image.shape) - image.shape[0]
    w_diff = max(image.shape) - image.shape[1]

    top = (h_diff + 1) // 2 if h_diff % 2 == 1 else h_diff // 2
    bottom = h_diff // 2
    left = (w_diff + 1) // 2 if w_diff % 2 == 1 else w_diff // 2    
    right = w_diff // 2
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_image, image.shape

def remove_pad(padded_image, original_shape):
    h_diff = padded_image.shape[0] - original_shape[0]
    w_diff = padded_image.shape[1] - original_shape[1]

    h_start = (h_diff + 1) // 2 if h_diff % 2 == 1 else h_diff // 2
    h_end = padded_image.shape[0] - h_diff // 2
    w_start = (w_diff + 1) // 2 if w_diff % 2 == 1 else w_diff // 2
    w_end = padded_image.shape[1] - w_diff // 2

    return padded_image[h_start:h_end, w_start:w_end]

class PennFudanDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)
        # Set all values greater-than or equal-to 1 to 255.
        mask = np.clip(mask, 0, 1)*255
        if self.transform or self.image_size:
            merged_image = np.concatenate((image, mask[:, :, None]), axis=2)
            merged_image, merged_shape = square_pad(merged_image)
            merged_image = transforms.ToTensor()(merged_image)
            if self.transform:
                merged_image = self.transform(merged_image)
            if self.image_size:
                merged_image = transforms.Resize((self.image_size, self.image_size))(merged_image)
            image = merged_image[:3, :, :]
            mask = merged_image[3, :, :].unsqueeze(0)
            return image, mask, merged_shape
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
        return image, mask
```

    C:\Users\Alqia\AppData\Local\Temp\ipykernel_23404\1208373077.py:6: DeprecationWarning: Sampling from a set deprecated
    since Python 3.9 and will be removed in a subsequent version.
      val_indices = random.sample(set(indices)-set(train_indices), k=int(len(indices)*0.1))
    

<a id='p1b'></a>
##### (b) Apply data augmentation to your dataset during training and show an example of your data augmentation in your report.

[Table of Contents](#Table-of-Contents)


```python
# Plot a subplot of the original and the augmented image and mask
train_dataset = PennFudanDataset(train_image_paths, train_mask_paths)
image, mask = train_dataset[0]
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
ax[0,0].imshow(image.permute(1, 2, 0))
ax[0,0].set_title("Original Image")
ax[0,1].imshow(mask.squeeze(), cmap="gray")
ax[0,1].set_title("Original Mask")
ax[0,2].imshow(image.permute(1, 2, 0))
ax[0,2].imshow(mask.squeeze(), alpha=0.5)
ax[0,2].set_title("Original Mask Overlayed on Image")

image_size = 256

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=image_size,scale=(0.35, 1.0), ratio=(1.0, 1.0))])
augmented_train_dataset = PennFudanDataset(
    train_image_paths, 
    train_mask_paths,
    transform=augmentation)
image, mask, _ = augmented_train_dataset[0]
ax[1,0].imshow(image.permute(1, 2, 0))
ax[1,0].set_title("Augmented Image")
ax[1,1].imshow(mask.squeeze(),cmap="gray")
ax[1,1].set_title("Augmented Mask")
ax[1,2].imshow(image.permute(1, 2, 0))
ax[1,2].imshow(mask.squeeze(), alpha=0.5)
ax[1,2].set_title("Original Mask Overlayed on Image")

batchsize = 8

test_dataset = PennFudanDataset(test_image_paths, test_mask_paths, image_size=image_size)

val_dataset = PennFudanDataset(val_image_paths, val_mask_paths, image_size=image_size)
```


    
![png](README1_files/README1_6_0.png)
    


<a id='p1c'></a>
##### (c) Implement and train a CNN for binary segmentation on your train split. Describe your network architecture, loss function, and any training hyper-parameters. You may implement any architecture you'd like, **but the implementation must be your own code.**

[Table of Contents](#Table-of-Contents)

#### Architecture

`x` (input) → `Conv_BN_ReLU1` → `Downsample1` → `x1` → `x2` → `Conv_BN_ReLU2` → `x3` → `Downsample2` → `x4` → `Conv_BN_ReLU3` → `x5` → `Upsample1` → `x6` → `cat(x3)` → `x7` → `Conv_BN_ReLU4` → `x8` → `Upsamle2` → `x9` → `cat(x1)` → `x10` → `Conv_BN_ReLU5` → `x11` → `conv6` → `x12` → `sigmoid` → `x13` (output)

![Architecture](UNET.png)


```python
class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.Conv_BN_ReLU1 = self._Conv_BN_ReLU(3, 16)
        self.Downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv_BN_ReLU2 = self._Conv_BN_ReLU(16, 32)
        self.Downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv_BN_ReLU3 = self._Conv_BN_ReLU(32, 32)
        self.Upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.Conv_BN_ReLU4 = self._Conv_BN_ReLU(64, 16)
        self.Upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.Conv_BN_ReLU5 = self._Conv_BN_ReLU(32, 16)
        self.Conv6 = nn.Conv2d(16, 1, kernel_size=1)

    def _Conv_BN_ReLU(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.Conv_BN_ReLU1(x) # 3x128x128 -> 16x128x128
        x2 = self.Downsample1(x1) # 16x128x128 -> 16x64x64
        x3 = self.Conv_BN_ReLU2(x2) # 16x64x64 -> 32x64x64
        x4 = self.Downsample2(x3) # 32x64x64 -> 32x32x32
        x5 = self.Conv_BN_ReLU3(x4) # 32x32x32 -> 32x32x32
        x6 = self.Upsample1(x5) # 32x32x32 -> 32x64x64
        x7 = torch.cat((x6, x3), dim=1) # 32x64x64 + 32x64x64 -> 64x64x64
        x8 = self.Conv_BN_ReLU4(x7) # 64x64x64 -> 16x64x64
        x9 = self.Upsample2(x8) # 16x64x64 -> 16x128x128
        x10 = torch.cat((x9, x1), dim=1) # 16x128x128 + 16x128x128 -> 32x128x128
        x11 = self.Conv_BN_ReLU5(x10) # 32x128x128 -> 16x128x128
        x12 = self.Conv6(x11) # 16x128x128 -> 1x128x128
        x13 = torch.sigmoid(x12) # 1x128x128 -> 1x128x128
        return x13
```

#### Loss Function

[Table of Contents](#Table-of-Contents)


```python
def dice_coefficient(output, ground_truth):
    numerical_stability = 1.
    output = output.view(-1)
    ground_truth = ground_truth.view(-1)
    intersection = (output * ground_truth).sum()
    return (2. * intersection + numerical_stability) / (output.sum() + ground_truth.sum() + numerical_stability)

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
    
    def forward(self, output, ground_truth):
        return 1 - dice_coefficient(output, ground_truth)
```

#### Training

[Table of Contents](#Table-of-Contents)


```python
def save_checkpoint(filename, model, optimizer, epoch, val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": val_loss}, 
        filename)
    
def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for image, mask, _ in train_loader:
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        pred = (model(image))
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for image, mask, _ in val_loader:
            image = image.to(device)
            mask = mask.to(device)
            pred = torch.round(model(image))
            loss = criterion(pred, mask)
            val_loss += loss.item()
    return val_loss / len(val_loader)


NUM_EPOCHS = 100
learning_rate=0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNET().to(device)
criterion = SoftDiceLoss()#nn.BCELoss()#
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, min_lr=1e-20,eps=1e-20,patience=10)

checkpoint_dir = "./checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

train_losses = []
val_losses = []
val_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
train_loader = DataLoader(augmented_train_dataset, batch_size=batchsize, shuffle=True)
```


```python
if os.path.exists(checkpoint_path):
    model, optimizer, epoch, val_loss = load_checkpoint(checkpoint_path, model, optimizer)
    print("Checkpoint loaded:\tstart epoch = {},\tval loss = {:.4f}".format(epoch, val_loss))
else:
    epoch = 0
    val_loss = np.inf
while val_loss > 0.35:
    epoch += 1
    progress_bar = tqdm(train_loader)
    train_loss = train(model, progress_bar, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print("Epoch: {}".format(epoch),
            "Train Loss: {:.4f}".format(train_loss),
            "Val Loss: {:.4f}".format(val_loss),
            "Learning rate: {}".format(optimizer.param_groups[0]['lr']),
            sep="\t")
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss == np.min(val_losses):
        save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss)
        print("Checkpoint saved:\tepoch = {},\tval loss = {:.4f}".format(epoch, val_loss))
```

    100%|██████████| 17/17 [00:05<00:00,  3.27it/s]
    

    Epoch: 1	Train Loss: 0.6674	Val Loss: 0.8449	Learning rate: 0.001
    Checkpoint saved:	epoch = 1,	val loss = 0.8449
    

    100%|██████████| 17/17 [00:04<00:00,  3.45it/s]
    

    Epoch: 2	Train Loss: 0.5865	Val Loss: 0.6495	Learning rate: 0.001
    Checkpoint saved:	epoch = 2,	val loss = 0.6495
    

    100%|██████████| 17/17 [00:05<00:00,  3.39it/s]
    

    Epoch: 3	Train Loss: 0.5657	Val Loss: 0.5340	Learning rate: 0.001
    Checkpoint saved:	epoch = 3,	val loss = 0.5340
    

    100%|██████████| 17/17 [00:05<00:00,  3.38it/s]
    

    Epoch: 4	Train Loss: 0.5371	Val Loss: 0.4970	Learning rate: 0.001
    Checkpoint saved:	epoch = 4,	val loss = 0.4970
    

    100%|██████████| 17/17 [00:05<00:00,  3.24it/s]
    

    Epoch: 5	Train Loss: 0.5226	Val Loss: 0.5647	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.31it/s]
    

    Epoch: 6	Train Loss: 0.4889	Val Loss: 0.5964	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.48it/s]
    

    Epoch: 7	Train Loss: 0.4958	Val Loss: 0.6220	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.13it/s]
    

    Epoch: 8	Train Loss: 0.4763	Val Loss: 0.5173	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.28it/s]
    

    Epoch: 9	Train Loss: 0.4638	Val Loss: 0.4713	Learning rate: 0.001
    Checkpoint saved:	epoch = 9,	val loss = 0.4713
    

    100%|██████████| 17/17 [00:05<00:00,  3.26it/s]
    

    Epoch: 10	Train Loss: 0.4416	Val Loss: 0.3687	Learning rate: 0.001
    Checkpoint saved:	epoch = 10,	val loss = 0.3687
    

    100%|██████████| 17/17 [00:04<00:00,  3.47it/s]
    

    Epoch: 11	Train Loss: 0.4217	Val Loss: 0.4542	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.50it/s]
    

    Epoch: 12	Train Loss: 0.4255	Val Loss: 0.3976	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.58it/s]
    

    Epoch: 13	Train Loss: 0.4169	Val Loss: 0.5168	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.49it/s]
    

    Epoch: 14	Train Loss: 0.3995	Val Loss: 0.5542	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.26it/s]
    

    Epoch: 15	Train Loss: 0.3978	Val Loss: 0.4363	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.46it/s]
    

    Epoch: 16	Train Loss: 0.3686	Val Loss: 0.3176	Learning rate: 0.001
    Checkpoint saved:	epoch = 16,	val loss = 0.3176
    


```python
if os.path.exists(checkpoint_path):
    model, optimizer, epoch, val_loss = load_checkpoint(checkpoint_path, model, optimizer)
    print("Checkpoint loaded:\tstart epoch = {},\tval loss = {:.4f}".format(epoch, val_loss))
else:
    epoch = 0
    val_loss = np.inf
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, min_lr=1e-20,eps=1e-20,patience=20)
while val_loss > 0.3:
    epoch += 1
    progress_bar = tqdm(train_loader)
    train_loss = train(model, progress_bar, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print("Epoch: {}".format(epoch),
            "Train Loss: {:.4f}".format(train_loss),
            "Val Loss: {:.4f}".format(val_loss),
            "Learning rate: {}".format(optimizer.param_groups[0]['lr']),
            sep="\t")
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss == np.min(val_losses):
        save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss)
        print("Checkpoint saved:\tepoch = {},\tval loss = {:.4f}".format(epoch, val_loss))
```

    Checkpoint loaded:	start epoch = 16,	val loss = 0.3176
    

    100%|██████████| 17/17 [00:04<00:00,  3.52it/s]
    

    Epoch: 17	Train Loss: 0.3650	Val Loss: 0.2626	Learning rate: 0.001
    Checkpoint saved:	epoch = 17,	val loss = 0.2626
    


```python
if os.path.exists(checkpoint_path):
    model, optimizer, epoch, val_loss = load_checkpoint(checkpoint_path, model, optimizer)
    print("Checkpoint loaded:\tstart epoch = {},\tval loss = {:.4f}".format(epoch, val_loss))
else:
    epoch = 0
    val_loss = np.inf
while val_loss > 0.25:
    epoch += 1
    progress_bar = tqdm(train_loader)
    train_loss = train(model, progress_bar, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print("Epoch: {}".format(epoch),
            "Train Loss: {:.4f}".format(train_loss),
            "Val Loss: {:.4f}".format(val_loss),
            "Learning rate: {}".format(optimizer.param_groups[0]['lr']),
            sep="\t")
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss == np.min(val_losses):
        save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss)
        print("Checkpoint saved:\tepoch = {},\tval loss = {:.4f}".format(epoch, val_loss))
```

    Checkpoint loaded:	start epoch = 17,	val loss = 0.2626
    

    100%|██████████| 17/17 [00:05<00:00,  2.98it/s]
    

    Epoch: 18	Train Loss: 0.3748	Val Loss: 0.2769	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.62it/s]
    

    Epoch: 19	Train Loss: 0.3599	Val Loss: 0.4063	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.47it/s]
    

    Epoch: 20	Train Loss: 0.3392	Val Loss: 0.3341	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.52it/s]
    

    Epoch: 21	Train Loss: 0.3254	Val Loss: 0.2400	Learning rate: 0.001
    Checkpoint saved:	epoch = 21,	val loss = 0.2400
    


```python
if os.path.exists(checkpoint_path):
    model, optimizer, epoch, val_loss = load_checkpoint(checkpoint_path, model, optimizer)
    print("Checkpoint loaded:\tstart epoch = {},\tval loss = {:.4f}".format(epoch, val_loss))
else:
    epoch = 0
    val_loss = np.inf
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, min_lr=1e-20,eps=1e-20,patience=20)
while val_loss > 0.2:
    epoch += 1
    progress_bar = tqdm(train_loader)
    train_loss = train(model, progress_bar, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print("Epoch: {}".format(epoch),
            "Train Loss: {:.4f}".format(train_loss),
            "Val Loss: {:.4f}".format(val_loss),
            "Learning rate: {}".format(optimizer.param_groups[0]['lr']),
            sep="\t")
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss == np.min(val_losses):
        save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss)
        print("Checkpoint saved:\tepoch = {},\tval loss = {:.4f}".format(epoch, val_loss))
```

    Checkpoint loaded:	start epoch = 21,	val loss = 0.2400
    

    100%|██████████| 17/17 [00:04<00:00,  3.47it/s]
    

    Epoch: 22	Train Loss: 0.3353	Val Loss: 0.3184	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.56it/s]
    

    Epoch: 23	Train Loss: 0.3407	Val Loss: 0.3700	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.43it/s]
    

    Epoch: 24	Train Loss: 0.3374	Val Loss: 0.2639	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.28it/s]
    

    Epoch: 25	Train Loss: 0.3039	Val Loss: 0.3233	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.34it/s]
    

    Epoch: 26	Train Loss: 0.3147	Val Loss: 0.3604	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.15it/s]
    

    Epoch: 27	Train Loss: 0.3158	Val Loss: 0.2482	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.37it/s]
    

    Epoch: 28	Train Loss: 0.3048	Val Loss: 0.2248	Learning rate: 0.001
    Checkpoint saved:	epoch = 28,	val loss = 0.2248
    

    100%|██████████| 17/17 [00:04<00:00,  3.71it/s]
    

    Epoch: 29	Train Loss: 0.2918	Val Loss: 0.3912	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.54it/s]
    

    Epoch: 30	Train Loss: 0.2903	Val Loss: 0.3248	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.58it/s]
    

    Epoch: 31	Train Loss: 0.3172	Val Loss: 0.3282	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.80it/s]
    

    Epoch: 32	Train Loss: 0.3021	Val Loss: 0.2289	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.68it/s]
    

    Epoch: 33	Train Loss: 0.3069	Val Loss: 0.2544	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.84it/s]
    

    Epoch: 34	Train Loss: 0.2774	Val Loss: 0.2756	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.72it/s]
    

    Epoch: 35	Train Loss: 0.2796	Val Loss: 0.2444	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.54it/s]
    

    Epoch: 36	Train Loss: 0.2887	Val Loss: 0.2966	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:05<00:00,  3.38it/s]
    

    Epoch: 37	Train Loss: 0.2796	Val Loss: 0.2472	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.66it/s]
    

    Epoch: 38	Train Loss: 0.2672	Val Loss: 0.2967	Learning rate: 0.001
    

    100%|██████████| 17/17 [00:04<00:00,  3.41it/s]
    

    Epoch: 39	Train Loss: 0.2683	Val Loss: 0.2163	Learning rate: 0.001
    Checkpoint saved:	epoch = 39,	val loss = 0.2163
    

    100%|██████████| 17/17 [00:04<00:00,  3.65it/s]
    

    Epoch: 40	Train Loss: 0.2463	Val Loss: 0.1891	Learning rate: 0.001
    Checkpoint saved:	epoch = 40,	val loss = 0.1891
    


```python
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
model, optimizer, start_epoch, val_loss = load_checkpoint(checkpoint_path, model, optimizer)
# Use the model on the test set and visualize the results
model.eval()
with torch.no_grad():
    image, mask, _ = next(iter(test_loader))
    image = image.to(device)
    mask = mask.to(device)
    pred = torch.round(model(image))
    pred = pred.cpu().numpy()
    mask = mask.cpu().numpy()
    image = image.cpu().numpy()
    pred = np.squeeze(pred, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(image[3].transpose(1, 2,0))
    ax[0].set_title("Image")
    ax[1].imshow(mask[3].squeeze(), cmap="gray")
    ax[1].set_title("Mask")
    ax[2].imshow(pred[3].squeeze(), cmap="gray")
    ax[2].set_title("Prediction")
    plt.show()
    

    for image in os.listdir("./out_of_distribution_images"):
        image = cv2.imread(os.path.join("./out_of_distribution_images", image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, shape = square_pad(image)
        image = cv2.resize(image, (image_size, image_size))
        image = image.transpose(2, 0, 1)
        image = image / 255
        image = torch.from_numpy(image).float()
        image = image.to(device)
        pred = model(image[None, ...])
        pred = torch.round(pred)
        pred = pred.cpu().numpy()
        pred = np.squeeze(pred, axis=1)
        plt.subplot(1, 2, 1)
        plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(pred[0], cmap="gray")
        plt.show()
```


    
![png](README1_files/README1_18_0.png)
    



    
![png](README1_files/README1_18_1.png)
    



    
![png](README1_files/README1_18_2.png)
    



    
![png](README1_files/README1_18_3.png)
    


The model is trained for 40 epochs and the results are shown below. The model is able to detect the edges of the objects in the image and the mask. The model is not able to detect the objects completely. This is because the dataset is very small and the model is not able to learn the features of the objects. The model is also not able to detect the objects that are not present in the training set.


<a id='p1d'></a>
##### (d) Report training loss, validation loss, and validation DICE curves. Comment on any overfitting or underfitting observed.

[Table of Contents](#Table-of-Contents)

<a id='p2a'></a>
##### (e) Report the average dice score over your test-set. **You should be able to achieve a score of around 0.7 or better**.

[Table of Contents](#Table-of-Contents)

<a id='p2b'></a>
##### (f) Show at least 3 example segmentations (i.e. show the RGB image, mask, and RGB image X mask for 3 samples) from your training data and 3 from your testing data. Comment on the generalization capabilities of your trained network.

[Table of Contents](#Table-of-Contents)

<a id='p2c'></a>
##### (g) Show at least 1 example segmentation on an input image **<ins>not</ins> from the FudanPed dataset**. Again, comment on the generalization capabilities of your network with respect to this "out-of-distribution" image.

[Table of Contents](#Table-of-Contents)
