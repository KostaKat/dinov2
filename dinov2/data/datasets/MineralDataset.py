import os
from .extended import ExtendedVisionDataset
from .preproccessing import MineralPreprocessing
from .decoders import ImageDataDecoder
import numpy as np
class MineralDataset(ExtendedVisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, **kwargs):
        # Ensure root and split are set as instance variables
        self.root = root
        self.split = split
        self.processor = MineralPreprocessing(transforms=False)  # Custom preprocessing
        self.image_paths = []

        # Build image paths based on train/val/test split
        split_dir = os.path.join(self.root, split)
        for dirpath, _, filenames in os.walk(split_dir):
            for file in filenames:
                if file.endswith(('jpg', 'jpeg', 'png', 'JPG')):
                    self.image_paths.append(os.path.join(dirpath, file))

        # Initialize parent class with transformations
        super().__init__(root=root, transform=transform, target_transform=target_transform, **kwargs)

    def get_image_data(self, index: int) -> bytes:
        """
        Get raw image data from the given index.
        """
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            return f.read()

    def get_target(self, index: int):
        """
        This dataset doesn't have labels, so it returns None.
        """
        return None

    def __getitem__(self, index: int):
        try:
            # Get image data
            image_data = self.get_image_data(index)

            # Decode the image and apply preprocessing
            img = ImageDataDecoder(image_data).decode()
            # make numpy array
            img = np.array(img)
            img = self.processor.preprocess_image(img)
            target = self.get_target(index)
            # Apply any provided transformations
            if self.transforms is not None:
                img, target = self.transforms(img, None)

            # Return processed image and None (as there are no labels)
            return img, target

        except Exception as e:
            # Add informative error message in case of an issue
            raise RuntimeError(f"Unable to read/process image at index {index}, path: {self.image_paths[index]}") from e

    def __len__(self) -> int:
        """
        Return the total number of image paths.
        """
        return len(self.image_paths)
