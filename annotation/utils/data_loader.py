"""
Data loader module for word recognition annotation tool.
Loads images from sorted_crops/ and corresponding labels from ground_truth/
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image


class DataLoader:
    """Loads word images and their ground truth labels."""
    
    def __init__(self, sorted_crops_dir: str = "sorted_crops", ground_truth_dir: str = "ground_truth"):
        """
        Initialize DataLoader.
        
        Args:
            sorted_crops_dir: Path to directory containing image subfolders
            ground_truth_dir: Path to directory containing label text files
        """
        self.sorted_crops_dir = Path(sorted_crops_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        
    def get_all_folders(self) -> List[str]:
        """Get list of all subfolder names in sorted_crops directory."""
        if not self.sorted_crops_dir.exists():
            return []
        
        folders = [f.name for f in self.sorted_crops_dir.iterdir() if f.is_dir()]
        return sorted(folders)
    
    def load_ground_truth(self, folder_name: str) -> List[str]:
        """
        Load ground truth labels for a specific folder.
        
        Args:
            folder_name: Name of the subfolder (e.g., "31")
            
        Returns:
            List of labels, one per line in the text file
        """
        gt_file = self.ground_truth_dir / f"{folder_name}.txt"
        
        if not gt_file.exists():
            return []
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        
        return labels
    
    def get_images_in_folder(self, folder_name: str) -> List[Path]:
        """
        Get sorted list of image paths in a specific folder.
        
        Args:
            folder_name: Name of the subfolder (e.g., "31")
            
        Returns:
            Sorted list of image file paths
        """
        folder_path = self.sorted_crops_dir / folder_name
        
        if not folder_path.exists():
            return []
        
        # Get all image files (jpg, jpeg, png)
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in folder_path.iterdir() 
                 if f.is_file() and f.suffix in image_extensions]
        
        # Sort by filename (e.g., 000.jpg, 001.jpg, etc.)
        return sorted(images)
    
    def load_all_data(self) -> List[Dict]:
        """
        Load all images with their corresponding ground truth labels.
        
        Returns:
            List of dictionaries with keys:
                - image_path: Path to image file
                - folder: Folder name
                - filename: Image filename
                - suggested_label: Corresponding ground truth label
                - index: Index within folder
        """
        all_data = []
        folders = self.get_all_folders()
        
        for folder in folders:
            images = self.get_images_in_folder(folder)
            labels = self.load_ground_truth(folder)
            
            # Match images to labels by index
            for idx, image_path in enumerate(images):
                # Get corresponding label if available
                suggested_label = labels[idx] if idx < len(labels) else ""
                
                all_data.append({
                    'image_path': str(image_path),
                    'folder': folder,
                    'filename': image_path.name,
                    'suggested_label': suggested_label,
                    'index': idx
                })
        
        return all_data
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and return a PIL Image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object or None if loading fails
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_folder_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about each folder.
        
        Returns:
            Dictionary mapping folder names to stats:
                - image_count: Number of images
                - label_count: Number of labels
                - match: Whether counts match
        """
        stats = {}
        folders = self.get_all_folders()
        
        for folder in folders:
            images = self.get_images_in_folder(folder)
            labels = self.load_ground_truth(folder)
            
            stats[folder] = {
                'image_count': len(images),
                'label_count': len(labels),
                'match': len(images) == len(labels)
            }
        
        return stats
