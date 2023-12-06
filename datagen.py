from keras.preprocessing.image import ImageDataGenerator

# Define the paths to the folders with images for training and validation
train_path = r"split/train"
val_path = r"split/val"

# Specify image dimensions and batch size
img_height = 128
img_width = 128
batch_size = 64


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
)

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    
)




validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
)

# Create the validation data generator
validation_generator = validation_datagen.flow_from_directory(
    val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    
)
