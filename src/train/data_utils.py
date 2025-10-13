from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from urllib.request import urlretrieve
import pandas as pd
from typing import Union, List, Optional
from transformers import AutoTokenizer, T5EncoderModel
import zarr
import torch
import numpy as np
import json 

class ZarrDataset(Dataset):
    """PyTorch Dataset for zarr arrays."""

    def __init__(self, zarr_path, array_key=None, max_samples=None, cache_size=8_388_608):
        """
        Initialize the dataset.

        Args:
            zarr_path (str): Path to the zarr file/directory
            array_key (str, optional): Key/name of the array if zarr contains multiple arrays
            max_samples (int, optional): Maximum number of samples to use (for testing/debugging)
            cache_size (int, optional): Number of samples to cache for batched loading
        """
        zarr_root = zarr.open(zarr_path, mode='r')
        self.cache_size = cache_size
        self.cache = None
        self.random_permutation = None
        self.cache_start_idx = -1
        self.cache_end_idx = -1

        # Handle different zarr structures
        if isinstance(zarr_root, zarr.Group):
            print(f"Zarr group found. Available arrays: {list(zarr_root.keys())}")

            if array_key is None:
                # If no key specified, try to find the array automatically
                arrays = [key for key in zarr_root.keys() if isinstance(zarr_root[key], zarr.Array)]
                if len(arrays) == 1:
                    array_key = arrays[0]
                    print(f"Using array: {array_key}")
                else:
                    raise ValueError(f"Multiple arrays found: {arrays}. Please specify array_key parameter.")

            self.zarr_array = zarr_root[array_key]

        elif isinstance(zarr_root, zarr.Array):
            # Single array case
            self.zarr_array = zarr_root

        else:
            raise ValueError(f"Unexpected zarr object type: {type(zarr_root)}")

        # Ensure the array has the expected shape
        assert len(self.zarr_array.shape) == 2, f"Expected 2D array, got shape {self.zarr_array.shape}"
        
        # Apply max_samples limit if specified
        self.total_samples = self.zarr_array.shape[0]
        if max_samples is not None and max_samples < self.total_samples:
            self.total_samples = max_samples
            print(f"Limiting dataset to {max_samples} samples (original: {self.zarr_array.shape[0]})")

        print(f"Loaded zarr array with shape: {self.zarr_array.shape}")
        print(f"Using {self.total_samples} samples")
        print(f"Array dtype: {self.zarr_array.dtype}")
        print(f"Array chunks: {self.zarr_array.chunks}")

    def __len__(self):
        """Return the total number of samples."""
        return self.total_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset with batched loading for better performance.

        Args:
            idx (int): Index of the sample

        Returns:
            torch.Tensor: Sample as a tensor
        """
        # Check if idx is in current cache
        if self.cache is None or idx < self.cache_start_idx or idx >= self.cache_end_idx:
            # Load a new batch of data
            start_idx = max(0, idx)
            end_idx = min(self.total_samples, start_idx + self.cache_size)
            
            # Load batch from zarr (much more efficient than individual row access)
            batch_data = self.zarr_array[start_idx:end_idx]

            # Convert to tensor and cache
            self.cache = torch.from_numpy(np.array(batch_data)).float()
            self.cache_start_idx = start_idx
            self.cache_end_idx = end_idx
            self.random_permutation = np.random.permutation(end_idx - start_idx)
        
        # Return sample from cache
        cache_idx = idx - self.cache_start_idx
        sample = self.cache[self.random_permutation[cache_idx]]
        return sample

class T5DataSet(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer_max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx:

        Returns:

        """
        prompt = self.data[idx]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        return text_inputs

def get_prompts_from_face_dataset():
    ds = load_dataset("OpenFace-CQUPT/HumanCaption-10M")
    prompts = ds['train']['human_caption']
    return prompts

def get_prompts_from_diffusion_db():
    import os
    if not os.path.exists('metadata.parquet'):
        table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
        urlretrieve(table_url, 'metadata.parquet')
    metadata_df = pd.read_parquet('metadata.parquet')
    prompts = metadata_df['prompt'].tolist()
    return prompts


def create_dataset(max_length:int =512, start_idx: Optional[int] = None, end_idx: Optional[int] = None,
                   tokenizer=None):
    prompts = get_prompts_from_face_dataset().extend(get_prompts_from_diffusion_db())
    if (start_idx is not None) and (end_idx is not None):
        prompts = prompts[start_idx:end_idx]
    return T5DataSet(prompts, max_length=max_length, tokenizer=tokenizer)
