# src/data/transforms.py
import torch
import torch.nn.functional as F
import random
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image):
        return (image - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return torch.flip(image, [2])
        return image


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return torch.flip(image, [1])
        return image


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(image, angle)
