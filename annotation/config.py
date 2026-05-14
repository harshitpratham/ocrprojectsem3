"""
Configuration file for the annotation tool.
"""

import os

# Application settings
APP_TITLE = "Word Recognition Annotation Tool"
APP_ICON = "📝"

# Data directories
SORTED_CROPS_DIR = "sorted_crops"
GROUND_TRUTH_DIR = "ground_truth"
ANNOTATIONS_DIR = "annotations"

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# UI settings
ITEMS_PER_PAGE = 10
DEFAULT_IMAGE_QUALITY = 95

# Export settings
EXPORT_DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

# Session settings
SESSION_TIMEOUT_MINUTES = 120

# Annotation settings
ENABLE_KEYBOARD_SHORTCUTS = True
AUTO_ADVANCE_ON_CORRECT = True
SHOW_ANNOTATION_HISTORY = True

# ============= SECURITY SETTINGS =============
# Admin creation key (change this or use environment variable)
ADMIN_CREATION_KEY = os.getenv('ADMIN_CREATION_KEY', 'admin_key_2024')

# Password requirements
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_NUMBERS = True

# Authentication settings
MAX_LOGIN_ATTEMPTS = 5
LOGIN_ATTEMPT_TIMEOUT_MINUTES = 15

# Hindi input support
ENABLE_HINDI_TRANSLITERATION = True
TRANSLITERATION_METHOD = 'google'  # 'google', 'indic', or 'hybrid'
HINGLISH_SUGGESTIONS = True

# Export formats
EXPORT_FORMATS = ['CSV', 'JSON', 'Excel', 'Parquet']
MAX_EXPORT_RECORDS = 100000

# Analytics
ENABLE_INTER_ANNOTATOR_AGREEMENT = True
QUALITY_THRESHOLD = 0.85  # 85% accuracy threshold

# Admin settings
MAX_PREVIEW_ITEMS = 10
ENABLE_QUALITY_REVIEW = True
