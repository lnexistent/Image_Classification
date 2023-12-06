from PIL import Image
import os
import csv
import cv2
import numpy as np

def check_image(file_path):
    #Check if an image file is readable
    try:
        with Image.open(file_path):
            pass
        return True
    except (OSError, IOError):
        return False

def count_images_by_extension(dataset_path):
    #Count the number of images with different extensions in each class folder
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)

        if os.path.isdir(class_path):
            #Count images with different extensions
            image_counts = {ext: sum(1 for img in os.listdir(class_path) if img.lower().endswith(ext)) for ext in ['.jpg', '.jpeg', '.png', '.bmp']}
            total_images = sum(image_counts.values())

            #Print the counts for each extension and total images in the class
            print(f"Class: {class_folder}")
            for ext, count in image_counts.items():
                print(f"  {ext} count: {count}")
            print(f"  Total images: {total_images}\n")

def clean_dataset(dataset_path):
    #Remove non-image files from the dataset
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)

        if os.path.isdir(class_path):
            #Find and remove files that are not valid images
            images_to_remove = [img for img in os.listdir(class_path) if not check_image(os.path.join(class_path, img))]
            for img in images_to_remove:
                os.remove(os.path.join(class_path, img))

def convert_all_to_jpg(dataset_path, output_csv_path):
    #Convert images to JPEG format and save conversion details to a CSV file
    converted_images = []
    reshaped_images = []

    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)

        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                _, img_extension = os.path.splitext(img_path)

                if img_extension.lower() not in ['.jpg', '.jpeg']:
                    #Convert the image to JPEG format
                    new_path = img_path.replace(img_extension, '.jpg')
                    Image.open(img_path).convert('RGB').save(new_path, 'JPEG')
                    converted_images.append((img_path, new_path))
                    os.remove(img_path)

                    #Preprocess the image and save the path and reshaped image to the reshaped CSV
                    reshaped_image_path = new_path.replace('.jpg', '_converted.jpg')
                    reshaped_image = preprocess_image(new_path)
                    Image.fromarray((reshaped_image * 255).astype(np.uint8)).save(reshaped_image_path, 'JPEG')
                    reshaped_images.append((new_path, reshaped_image_path))

    #Write the information to the converted CSV file
    with open(output_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Original File', 'Converted File'])
        csv_writer.writerows(converted_images)

def preprocess_image(image_path, target_size=(128, 128)):
    #Preprocess images by resizing and cropping
    #Read the image
    image = cv2.imread(image_path)

    #Resize while maintaining the aspect ratio
    h, w, _ = image.shape
    aspect_ratio = w / h

    if aspect_ratio >= 1:  
        new_w = target_size[0]
        new_h = int(new_w / aspect_ratio)
    else:  
        new_h = target_size[1]
        new_w = int(new_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))

    #Crop the image to match the target size
    crop_top = max(0, (new_h - target_size[1]) // 2)
    crop_bottom = min(new_h, crop_top + target_size[1])
    crop_left = max(0, (new_w - target_size[0]) // 2)
    crop_right = min(new_w, crop_left + target_size[0])

    cropped_image = resized_image[crop_top:crop_bottom, crop_left:crop_right]

    #Ensure the final image has the target size
    final_image = cv2.resize(cropped_image, target_size)

    #Normalize pixel values
    normalized_image = final_image / 255.0

    return normalized_image


def reshape_all_images(dataset_path, reshaped_csv_path='reshaped_images.csv', target_shape=(128, 128)):
    #Reshape images to a specified size and save details to a CSV file
    reshaped_images = []

    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)

        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                #reshaped_image_path = img_path.replace('.jpg', '_reshaped.jpg')
                
                #Check if the image needs reshaping
                original_image = cv2.imread(img_path)
                reshaped_image = preprocess_image(img_path, target_shape)
                
                if original_image.shape[:2] != reshaped_image.shape[:2]:
                    #Save the reshaped image with the original filename
                    Image.fromarray((reshaped_image * 255).astype(np.uint8)).save(img_path, 'JPEG')
                    reshaped_images.append((img_path, original_image.shape[:2]))

    #Write the information to the reshaped CSV file
    with open(reshaped_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Original File', 'Original Size'])
        csv_writer.writerows(reshaped_images)

def check_corrupted_images(dataset_path):
    #Check for corrupted images in the dataset
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)

        if os.path.isdir(class_path):
            corrupted_count = sum(1 for img in os.listdir(class_path) if not check_image(os.path.join(class_path, img)))

            print(f"Class: {class_folder}")
            print(f"  Corrupted images count: {corrupted_count}\n")

    #Manually specify four files to be removed
    files_to_remove = ["00000624.jpg", "00000683.jpg"]

    for file_to_remove in files_to_remove:
        file_path = os.path.join(dataset_path, file_to_remove)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed manually specified file: {file_to_remove}")

def process_images(main_dataset_path, output_csv_path='converted_images.csv', reshaped_csv_path='reshaped_images.csv', target_shape=(128, 128)):
    #Automate the entire image processing pipeline
    
    #Check image extensions
    count_images_by_extension(main_dataset_path)

    #Clean the dataset
    clean_dataset(main_dataset_path)

    #Convert all images to .jpg and resize to a common shape
    convert_all_to_jpg(main_dataset_path, output_csv_path)

    #Reshape images and save details to a separate CSV
    reshape_all_images(main_dataset_path, reshaped_csv_path, target_shape)

    #Check for corrupted images after conversion
    check_corrupted_images(main_dataset_path)

    #Print the information about converted images
    converted_images_count = len(open(output_csv_path).readlines()) - 1
    print(f"Total images converted: {converted_images_count}")
