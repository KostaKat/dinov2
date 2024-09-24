import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
class MineralPreprocessing:
    """
    A simplified preprocessing class for mineral images that focuses on using
    L*a*b* color space for contrast enhancement and converting back to RGB for the Swin Transformer.
    """

    def __init__(self, transforms=True):
        """
        Initializes the class with predefined normalization values.
        """
        self.mean = [0.3373, 0.3125, 0.3718]  # Standard mean values for normalization (ImageNet)
        self.std = [0.1334, 0.1314, 0.1472]   # Standard deviation for normalization (ImageNet)
        self.transforms = transforms
        if transforms:
            # Albumentations pipeline for basic normalization and tensor conversion
            self.albumentations_transform = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),  # Normalize to match pretrained models
                ToTensorV2()  # Convert to PyTorch tensor
            ])

    def convert_to_lab(self, img: np.ndarray) -> np.ndarray:
        """
        Converts the image from BGR to L*a*b* color space and enhances contrast
        by applying CLAHE to the L-channel.
        """
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(img_lab)

        # Apply CLAHE to the L-channel for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Merge the enhanced L-channel back with A and B channels
        img_lab_clahe = cv2.merge((l_clahe, a, b))

        # Convert back to BGR (or RGB) after contrast enhancement
        enhanced_img = cv2.cvtColor(img_lab_clahe, cv2.COLOR_Lab2BGR)

        return enhanced_img

    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocesses the image by converting it to L*a*b* color space for contrast enhancement,
        and then converting it back to BGR/RGB for the transformer model.
        """
        # Convert to L*a*b* and apply CLAHE
        img_enhanced = self.convert_to_lab(img)
        img_tensor = img_enhanced
       
        return  Image.fromarray(img_enhanced)

    def unnormalize(self, tensor_img: torch.Tensor) -> torch.Tensor:
        """
        Reverses normalization for visualization.
        """
        tensor_img = tensor_img.clone()

        for t, m, s in zip(tensor_img, self.mean, self.std):
            t = t.mul(s).add(m)  # Reverse normalization

        return tensor_img
