import cv2
import os

# Define input and output directories
# input_img_dir = 'EvaluationSoftware/testing_dataset/04_GT/SEG/'
input_img_dir = '2-GT/imgs'
# input_mask_dir = 'EvaluationSoftware/testing_dataset/04_GT/TRA/'
input_mask_dir = '2-GT/mask'
output_img_dir = '2-GT/imgs'
output_mask_dir = '2-GT/mask'

# Create output directories if they don't exist
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Get the list of image file names
image_files = os.listdir(input_img_dir)

# Define the target size
new_width = 1088
new_height = 672
target_size = (new_width, new_height)

for image_file in image_files:
    # Load the image
    image_path = os.path.join(input_img_dir, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load image as is, without color changes

    # Resize the image using bilinear interpolation
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Save the resized image
    output_image_path = os.path.join(output_img_dir, image_file)
    cv2.imwrite(output_image_path, resized_image)

    # Load the corresponding mask
    mask_file = os.path.splitext(image_file)[0] + '.tif'
    mask_path = os.path.join(input_mask_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # Load mask as is, without color changes

    # Resize the mask using nearest-neighbor interpolation
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Save the resized mask
    output_mask_path = os.path.join(output_mask_dir, mask_file)
    cv2.imwrite(output_mask_path, resized_mask)

print("Resizing and saving complete!")
