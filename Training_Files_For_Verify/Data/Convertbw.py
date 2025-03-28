import os
import cv2

# Define source and destination directories
source_dir = r'convert'  # Replace with your actual "convert" folder path
output_dir = r'convert_bw'  # New folder for black-and-white images

# Supported image extensions
valid_extensions = ('.jpg', '.jpeg', '.png')

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
else:
    print(f"Output directory already exists: {output_dir}")

# Check if source directory exists
if not os.path.exists(source_dir):
    print(f"Source directory {source_dir} does not exist. Exiting...")
    exit()

# Get list of images in the source directory
img_names = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)]
print(f"Found {len(img_names)} images in {source_dir}: {img_names}")

# Process each image
for img_name in img_names:
    # Construct full paths
    src_path = os.path.join(source_dir, img_name)
    dest_path = os.path.join(output_dir, f"bw_{img_name}")  # Prefix "bw_" to filename

    # Load the image
    img = cv2.imread(src_path)
    if img is None:
        print(f"Failed to load image: {src_path}")
        continue

    # Convert to grayscale (black and white)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite(dest_path, gray_img)
    print(f"Converted and saved: {dest_path}")

print("Conversion complete!")