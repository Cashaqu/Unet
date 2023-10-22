import torch
import torchvision.transforms as transforms
from PIL import Image
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_imgs, path_to_masks, trans=None):
        self.img_names = sorted([path_to_imgs + filename for filename in os.listdir(path_to_imgs)])
        self.mask_names = sorted([path_to_masks + filename for filename in os.listdir(path_to_masks)])
        self.trans = trans

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image_path = self.img_names[index]
        mask_path = self.mask_names[index]
        img = Image.open(image_path)
        mask = Image.open(mask_path)
        img_tensor = self.trans(img)
        mask_tensor = self.trans(mask)
        return img_tensor, mask_tensor


def to_loader(path_X, path_y, batch_size):
    transformation = transforms.Compose([transforms.Grayscale(1),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor()])
    dataset = CustomDataset(path_X, path_y, trans=transformation)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)