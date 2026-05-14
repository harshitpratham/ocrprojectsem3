"""
Storage module for annotation tool.
Handles user management and annotation persistence with full history.
"""

import os
import json
import csv
import hashlib
import secrets
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from config import PASSWORD_MIN_LENGTH, PASSWORD_REQUIRE_UPPERCASE, PASSWORD_REQUIRE_NUMBERS, ADMIN_CREATION_KEY


class AnnotationStorage:
    """Manages annotation storage with history tracking."""
    
    def __init__(self, annotations_dir: str = "annotations"):
        """
        Initialize AnnotationStorage.
        
        Args:
            annotations_dir: Base directory for all annotation data
        """
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(exist_ok=True)
        
        self.users_file = self.annotations_dir / "users.json"
        self.history_csv = self.annotations_dir / "history.csv"
        self.history_json = self.annotations_dir / "history.json"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize storage files if they don't exist."""
        # Initialize users file
        if not self.users_file.exists():
            with open(self.users_file, 'w') as f:
                json.dump([], f)
        
        # Initialize history CSV
        if not self.history_csv.exists():
            with open(self.history_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'annotation_id', 'image_path', 'folder', 'filename',
                    'suggested_label', 'is_correct', 'corrected_label',
                    'annotator', 'timestamp', 'status', 'confidence_tag',
                ])
                writer.writeheader()
        
        # Initialize history JSON
        if not self.history_json.exists():
            with open(self.history_json, 'w') as f:
                json.dump([], f)
    
    def register_user(self, username: str, password: str, role: str = "annotator", admin_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Register a new user with password authentication.
        
        Args:
            username: Username
            password: Password
            role: User role ("annotator" or "admin")
            admin_key: Required if role is "admin"
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        users = self.load_users()
        
        # Validate username
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if any(u['username'] == username for u in users):
            return False, "Username already exists"
        
        # Validate password
        pwd_validation = self._validate_password(password)
        if not pwd_validation[0]:
            return False, pwd_validation[1]
        
        # Admin role requires admin key
        if role == "admin":
            if not admin_key or admin_key != ADMIN_CREATION_KEY:
                return False, "Invalid admin creation key"
        
        # Hash password
        password_hash = self._hash_password(password)
        
        # Add new user
        users.append({
            'username': username,
            'password_hash': password_hash,
            'role': role,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True
        })
        
        self._save_users(users)
        return True, f"User '{username}' created successfully as {role}"
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (success: bool, user_dict: Optional[Dict])
        """
        users = self.load_users()
        
        for user in users:
            if user['username'] == username:
                if not user.get('is_active', True):
                    return False, None
                
                if self._verify_password(password, user.get('password_hash', '')):
                    # Update last login
                    user['last_login'] = datetime.now().isoformat()
                    self._save_users(users)
                    return True, user
                else:
                    return False, None
        
        return False, None
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password with salt using PBKDF2."""
        salt = secrets.token_hex(32)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"
    
    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, pwd_hash = password_hash.split('$')
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return new_hash.hex() == pwd_hash
        except:
            return False
    
    @staticmethod
    def _validate_password(password: str) -> Tuple[bool, str]:
        """Validate password strength."""
        if len(password) < PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
        
        if PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if PASSWORD_REQUIRE_NUMBERS and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        
        return True, "Password valid"
    
    def disable_user(self, username: str) -> bool:
        """Deactivate a user account (keep annotations)."""
        users = self.load_users()
        for user in users:
            if user['username'] == username:
                user['is_active'] = False
                self._save_users(users)
                return True
        return False
    
    def enable_user(self, username: str) -> bool:
        """Reactivate a user account."""
        users = self.load_users()
        for user in users:
            if user['username'] == username:
                user['is_active'] = True
                self._save_users(users)
                return True
        return False
    
    def delete_user(self, username: str) -> bool:
        """Delete a user account (keep annotations)."""
        users = self.load_users()
        users = [u for u in users if u['username'] != username]
        self._save_users(users)
        return True
    
    def update_password(self, username: str, new_password: str) -> Tuple[bool, str]:
        """Update user password."""
        pwd_validation = self._validate_password(new_password)
        if not pwd_validation[0]:
            return False, pwd_validation[1]
        
        users = self.load_users()
        for user in users:
            if user['username'] == username:
                user['password_hash'] = self._hash_password(new_password)
                self._save_users(users)
                return True, "Password updated successfully"
        
        return False, "User not found"
    
    def load_users(self) -> List[Dict]:
        """Load all registered users."""
        with open(self.users_file, 'r') as f:
            return json.load(f)
    
    def _save_users(self, users: List[Dict]):
        """Save users to file."""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        users = self.load_users()
        for user in users:
            if user['username'] == username:
                return user
        return None
    
    def save_annotation(self, annotation: Dict):
        """
        Save a single annotation to history (append-only).
        
        Args:
            annotation: Dictionary containing:
                - image_path: Path to image
                - folder: Folder name
                - filename: Image filename
                - suggested_label: Original suggested label
                - is_correct: Boolean indicating if label is correct
                - corrected_label: Corrected label if is_correct is False
                - annotator: Username of annotator
        """
        # Generate unique annotation ID
        annotation_id = self._generate_annotation_id()
        
        # Add metadata
        csv_fields = [
            'annotation_id', 'image_path', 'folder', 'filename',
            'suggested_label', 'is_correct', 'corrected_label',
            'annotator', 'timestamp', 'status', 'confidence_tag',
        ]
        full_annotation = {
            'annotation_id': annotation_id,
            'timestamp': datetime.now().isoformat(),
            **annotation,
        }
        for k in csv_fields:
            full_annotation.setdefault(k, '')

        # Append to CSV
        with open(self.history_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
            writer.writerow(full_annotation)
        
        # Append to JSON
        history = self.load_history()
        history.append(full_annotation)
        with open(self.history_json, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _generate_annotation_id(self) -> str:
        """Generate a unique annotation ID."""
        history = self.load_history()
        return f"ANN_{len(history) + 1:06d}"
    
    def load_history(self) -> List[Dict]:
        """Load all annotation history."""
        with open(self.history_json, 'r') as f:
            return json.load(f)
    
    def load_history_df(self) -> pd.DataFrame:
        """Load annotation history as pandas DataFrame from JSON (more reliable than CSV)."""
        try:
            history = self.load_history()
            if not history:
                return pd.DataFrame()
            df = pd.DataFrame(history)
            # Parse timestamp with error handling
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        except Exception as e:
            print(f"Error loading history: {e}")
            return pd.DataFrame()
    
    def get_user_annotations(self, username: str) -> List[Dict]:
        """Get all annotations by a specific user."""
        history = self.load_history()
        return [ann for ann in history if ann['annotator'] == username]
    
    def get_user_annotations_df(self, username: str) -> pd.DataFrame:
        """Get user annotations as DataFrame."""
        df = self.load_history_df()
        if df.empty:
            return df
        return df[df['annotator'] == username]
    
    def get_annotated_images(self, username: str) -> set:
        """Get set of image paths that user has already annotated."""
        annotations = self.get_user_annotations(username)
        return {ann['image_path'] for ann in annotations}
    
    def get_user_stats(self, username: str) -> Dict:
        """
        Get statistics for a specific user.
        
        Returns:
            Dictionary with stats:
                - total: Total annotations
                - correct: Number marked as correct
                - incorrect: Number marked as incorrect
                - correct_percentage: Percentage marked correct
        """
        annotations = self.get_user_annotations(username)
        
        if not annotations:
            return {
                'total': 0,
                'correct': 0,
                'incorrect': 0,
                'correct_percentage': 0.0
            }
        
        correct = sum(1 for ann in annotations if ann['is_correct'])
        incorrect = len(annotations) - correct
        
        return {
            'total': len(annotations),
            'correct': correct,
            'incorrect': incorrect,
            'correct_percentage': (correct / len(annotations) * 100) if annotations else 0.0
        }
    
    def get_all_user_stats(self) -> List[Dict]:
        """Get statistics for all users."""
        users = self.load_users()
        stats = []
        
        for user in users:
            username = user['username']
            user_stats = self.get_user_stats(username)
            stats.append({
                'username': username,
                'role': user['role'],
                **user_stats
            })
        
        return stats
    
    def export_to_csv(self, filepath: str, username: Optional[str] = None):
        """
        Export annotations to CSV file.
        
        Args:
            filepath: Output file path
            username: If provided, export only this user's annotations
        """
        if username:
            df = self.get_user_annotations_df(username)
        else:
            df = self.load_history_df()
        
        df.to_csv(filepath, index=False)
    
    def export_to_json(self, filepath: str, username: Optional[str] = None):
        """
        Export annotations to JSON file.
        
        Args:
            filepath: Output file path
            username: If provided, export only this user's annotations
        """
        if username:
            data = self.get_user_annotations(username)
        else:
            data = self.load_history()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_latest_annotation_for_image(self, image_path: str, username: str) -> Optional[Dict]:
        """Get the most recent annotation for a specific image by a user."""
        annotations = self.get_user_annotations(username)
        user_image_annotations = [ann for ann in annotations if ann['image_path'] == image_path]
        
        if not user_image_annotations:
            return None
        
        # Sort by timestamp and return latest
        return sorted(user_image_annotations, key=lambda x: x['timestamp'], reverse=True)[0]
