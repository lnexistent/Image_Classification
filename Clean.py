from Cleaning_Stuff import process_images
from shape import check_shape



main_dataset_path = '8-Objects-Training'
output_csv_path = 'converted_images.csv'
reshaped_csv_path = 'reshaped_images.csv'

process_images(main_dataset_path, output_csv_path, reshaped_csv_path)
check_shape(main_dataset_path)
