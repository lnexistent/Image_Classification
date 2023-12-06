import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to your original dataset
original_dataset_path = r'8-Objects-Training'

# Set the path to create a new base directory for the split dataset
base_dir = r'split'
os.makedirs(base_dir, exist_ok=True)

# Set the names for the train, validation, and test folders
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')  # New directory for validation
test_dir = os.path.join(base_dir, 'test')

# Create the train, validation, and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each class folder in the original dataset
for class_folder in os.listdir(original_dataset_path):
    class_path = os.path.join(original_dataset_path, class_folder)

    # Split the images into train (60%), validation (20%), and test (20%) sets
    train_images, temp_images = train_test_split(os.listdir(class_path), test_size=0.2, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    # Create class directories in the train, validation, and test sets
    train_class_dir = os.path.join(train_dir, class_folder)
    val_class_dir = os.path.join(val_dir, class_folder)
    test_class_dir = os.path.join(test_dir, class_folder)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy images to the respective directories
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))
