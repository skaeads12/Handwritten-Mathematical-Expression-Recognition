
import json
import albumentations as A
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

class MERecognitionDataset(Dataset):

    def __init__(
        self,
        data_path: str = None,
        image_dir: str = None,
        transform: A.Compose = None,
        sequence_length: int = None,
    ):
        self.data = json.load(open(data_path, 'r'))
        self.image_dir = image_dir
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def get_dataloader(self, batch_size: int = None, shuffle: bool = None):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def preprocess(self, x):
        x = x.replace("$", "")
        x = x.replace(" ", "")

        return x

    def __getitem__(self, index):
        sample = self.data[index]

        image_path = self.image_dir + sample["images"]
        image = cv2.imread(image_path)
        image = self.transform(image=image)["image"]

        labels = sample["labels"]
        labels = self.preprocess(labels)

        labels = "\x02" + labels + "\x03"
        attention_mask = [1] * len(labels)

        if len(labels) < self.sequence_length:
            labels += "\x00" * (self.sequence_length - len(labels))
            attention_mask += [0] * (self.sequence_length - len(attention_mask))
        else:
            labels = labels[:self.sequence_length]
            attention_mask = attention_mask[:self.sequence_length]

        label_ids = torch.tensor([ord(i) for i in labels])
        attention_mask = torch.LongTensor(attention_mask)
        
        return {"images": image, "label_ids": label_ids, "attention_mask": attention_mask}
