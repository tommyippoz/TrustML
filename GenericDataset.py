import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
import pandas as pd
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initializes the CustomDataset.

        Parameters:
        - data_dir: Path to the directory containing images.
        - transform: Transformations to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        # Load the dataset with ImageFolder
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        # Store class names
        self.classes = self.dataset.classes

    def __getitem__(self, idx):
        """
        Retrieves an image and its label from the dataset at the given index.

        Parameters:
        - idx: Index of the image.

        Returns:
        - image: The transformed image.
        - labels: The label index of the image.
        """
        image, labels = self.dataset[idx]
        return image, labels

    def __len__(self):
        return len(self.dataset)

class CustomCSVDataset(Dataset):
    def __init__(self, csv_file=None, data_frame=None, data_dir='', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if data_frame is None:
            self.annotations = pd.read_csv(csv_file)
        else:
            self.annotations = data_frame
        self.root_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.annotations.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        return image, label

class GenericDatasetLoader:
    def __init__(self, dataset_name=None, root_dir='', batch_size=1, csv_file=None, data_frame=None, **kwargs):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.csv_file = csv_file
        self.data_frame = data_frame
        self.kwargs = kwargs

    def load_dataset(self, transform, split='train'):
        if self.dataset_name == 'CIFAR10':
            return self.load_cifar(split)
        elif self.dataset_name == 'MNIST':
            return self.load_mnist(split)
        elif self.dataset_name == 'FLOWER':
            return self.load_flower(split)
        elif self.dataset_name == 'CUSTOM':
            return self.load_custom(split,transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def load_cifar(self, transform, split='train'):
        return datasets.CIFAR10(root=self.root_dir, train=(split == 'train'), transform=transform, download=True)

    def load_mnist(self, transform, split='train'):
        return datasets.MNIST(root=self.root_dir, train=(split == 'train'), transform=transform, download=True)

    def load_flower(self, transform, split='train'):
        split_type = 'train' if split == 'train' else 'test'
        return datasets.Flowers102(root=self.root_dir, split=split_type, transform=transform, download=True)

    def load_custom(self, transform, split='train'):
        if self.csv_file is None and self.data_frame is None:
            dataset_dir = os.path.join(self.root_dir, split)
            return CustomDataset(data_dir=dataset_dir, transform=transform)
        else:
            return CustomCSVDataset(csv_file=self.csv_file, data_frame=self.data_frame, data_dir=self.root_dir, transform=transform)

    def create_dataloader(self, transform, split='train',shuffle = True):
        dataset = self.load_dataset(split,transform)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=2,persistent_workers=False)

    def get_num_channels(self, dataloader):
        """
        Determines the number of channels in the images from a DataLoader.

        Args:
            dataloader (torch.utils.data.DataLoader): A DataLoader containing image data.

        Returns:
            int: Number of channels in the images.
        """
        # Iterate through the dataloader to get one batch of images
        for images, _ in dataloader:
            # Get the shape of the first image in the batch (assuming format [batch, channels, height, width])
            num_channels = images.shape[1]
            return num_channels

    def extract_labels(self, dataloader):
        """
        Extracts labels from a DataLoader.

        Args:
            dataloader (torch.utils.data.DataLoader): The DataLoader from which to extract labels.

        Returns:
            np.array: A NumPy array containing all labels from the DataLoader.
        """
        labels_list = []

        # Iterate over the DataLoader
        for _, labels in dataloader:
            # Append labels to the list
            labels_list.extend(labels.numpy())  # Convert tensor to numpy array and extend the list

        # Convert the list to a numpy array
        return np.array(labels_list)

    def dataloader_to_dataframe(self, dataloader):
        data = []
        for inputs, targets in dataloader:
            inputs_flat = inputs.view(inputs.size(0), -1).numpy()
            targets = targets.numpy()
            for img, label in zip(inputs_flat, targets):
                data.append({'Flattened Image': img, 'Label': label})
        df = pd.DataFrame(data)
        df.to_csv('SVM_abnormal.csv', index=False)
        return df
