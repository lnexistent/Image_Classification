from PIL import Image
import os

def check_shape(dataset_path):
    def get_image_shapes_count(dataset_path):
        image_shapes_count = {}

        for class_folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_folder)

            if os.path.isdir(class_path):
                for img in os.listdir(class_path):
                    img_path = os.path.join(class_path, img)

                    try:
                        with Image.open(img_path) as img_obj:
                            width, height = img_obj.size
                            shape = (width, height)

                            if shape in image_shapes_count:
                                image_shapes_count[shape] += 1
                            else:
                                image_shapes_count[shape] = 1
                    except (OSError, IOError):
                        print(f"Skipping {img_path} as it cannot be opened.")

        return image_shapes_count

    # Get image shapes and count
    image_shapes_count = get_image_shapes_count(dataset_path)

    # Print the count of files for each shape
    print("Count of files for each shape:")
    for shape, count in image_shapes_count.items():
        print(f"Shape: {shape}, Count: {count}")


