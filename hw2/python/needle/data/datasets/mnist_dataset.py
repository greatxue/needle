from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms if transforms else None
        with gzip.open(image_filename, 'rb') as img_file:
            magic_num, img_num, row, col = struct.unpack('>iiii', img_file.read(16)) 
            img_size = row * col
        
            imgs = [np.array(struct.unpack(f"{img_size}B", img_file.read(img_size)), dtype=np.float32) 
                for _ in range(img_num)] 
            imgs = np.vstack(imgs) 
            self.X = (imgs - np.min(imgs)) / np.max(imgs) 
    
        with gzip.open(label_filename, "rb") as label_file:
            magic_num, label_num = struct.unpack(">ii", label_file.read(8)) 
            self.y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8) 
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X, y = self.X[index], self.y[index]
        if self.transforms:
            X_in = X.reshape(28, 28, -1)
            X_out = self.apply_transforms(X_in)
            X_ret = X_out.reshape(-1, 28 * 28)
            return X_ret, y
        else:
            return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION