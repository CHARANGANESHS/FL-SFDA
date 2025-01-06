import os
from torch.utils.data import Dataset
from PIL import Image

class GlaSClassificationDataset(Dataset):
    """
    Minimal classification dataset for GlaS. 
    Ignores masks, focusing on images + Grade.csv.
    """
    def __init__(self, image_dir, grade_dict, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        for fname in os.listdir(image_dir):
            base, _ = os.path.splitext(fname)
            if base in grade_dict:
                self.samples.append(fname)
        self.grade_dict = grade_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        base, _ = os.path.splitext(fname)
        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert('RGB')
        label = self.grade_dict[base]
        if self.transform:
            image = self.transform(image)
        return image, label, base