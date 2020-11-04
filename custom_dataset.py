import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

# class ToTensor(object):
#     def __call__(self, data):
#         image, label = data.iloc[:, 0], data.iloc[:,1]


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, img_path, transform=None):

        self.img_path = img_path

        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
       
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(self.img_path + '/' + single_image_name)
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = torch.from_numpy(np.array([self.label_arr[index]]))

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
