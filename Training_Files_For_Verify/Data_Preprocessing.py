import cv2
import os
import numpy as np
from sklearn.utils import shuffle

# Provide your data path
data_path = r'Data'

# Categories: True (Aadhar), False (Non-Aadhar)
categories = ["True", "False"]
labels = [1, 0]  # 1 for True (Aadhar), 0 for False (Non-Aadhar)
label_dict = dict(zip(categories, labels))

img_size = 100

data = []
target = []

# Traverse through True and False folders
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        print(img_path)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))

        data.append(resized)
        target.append(label_dict[category])

# Normalize our data
data = np.array(data) / 255.0
data = np.reshape(data, (-1, img_size, img_size, 1))

target = np.array(target)

# Shuffle data
data, target = shuffle(data, target, random_state=42)

# Save our data
np.save('data', data)
np.save('target', target)

print(f"Processed {len(data)} images: {sum(target)} True, {len(target) - sum(target)} False")