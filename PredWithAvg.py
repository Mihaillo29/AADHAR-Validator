import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model

# Load models
yolo_model = YOLO("best.pt")  # YOLOv8 model for cropping/detection
tf_model = load_model("Card_Detector_ResNet.keras")  # TensorFlow model for classification

# Labels for binary classification
labels_dict = {1: 'Aadhar Card', 0: 'Not an Aadhar Card'}

class AadhaarCropAndPredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aadhaar Crop & Predict Tool")
        self.root.geometry("800x600")

        # Variables
        self.image_path = None
        self.processed_image = None

        # GUI Elements
        self.upload_btn = tk.Button(root, text="Upload and Process Image", command=self.upload_and_process)
        self.upload_btn.pack(pady=10)

        self.save_btn = tk.Button(root, text="Save Result", command=self.save_image, state="disabled")
        self.save_btn.pack(pady=10)

        self.canvas = tk.Canvas(root, width=700, height=400)
        self.canvas.pack(pady=20)

        self.label = tk.Label(root, text="", font=("Arial", 12))
        self.label.pack(pady=10)

    def adjust_orientation(self, img):
        """Check and rotate image to horizontal if vertical."""
        height, width = img.shape[:2]
        if height > width:  # Vertical image
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def preprocess_for_model(self, image):
        """Convert image to grayscale, resize, normalize, and reshape for model input."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_image, (100, 100))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 100, 100, 1)
        return reshaped

    def upload_and_process(self):
        """Upload image, adjust orientation, crop with YOLOv8 (or use whole image if no region detected),
        apply flips, and predict with TensorFlow."""
        # Upload image
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not self.image_path:
            print("No image selected!")
            return

        # Load original image
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"Could not load image at {self.image_path}")
            return

        # Adjust orientation to horizontal
        img = self.adjust_orientation(img)

        # Step 1: Detect regions with YOLOv8
        results = yolo_model(img)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Format: [x_min, y_min, x_max, y_max]

        # If no region is detected, use the entire image for verification
        if len(boxes) == 0:
            print("No regions detected! Using the full image for verification.")
            cropped_result = img
        else:
            # Determine the bounding box that encloses all detected regions
            x_min = int(np.min(boxes[:, 0]))
            y_min = int(np.min(boxes[:, 1]))
            x_max = int(np.max(boxes[:, 2]))
            y_max = int(np.max(boxes[:, 3]))
            cropped_result = img[y_min:y_max, x_min:x_max]
            print(f"Detected region cropped: ({x_min}, {y_min}) to ({x_max}, {y_max})")

        # Create augmented versions: original, vertical flip, horizontal flip, and both axes flipped
        images = {
            "original": cropped_result,
            "flip_vertical": cv2.flip(cropped_result, 0),
            "flip_horizontal": cv2.flip(cropped_result, 1),
            "flip_both": cv2.flip(cropped_result, -1)
        }

        predictions = []
        # Process each augmented version and get the model's prediction
        for key, im in images.items():
            processed = self.preprocess_for_model(im)
            pred = tf_model.predict(processed)[0][0]  # Assuming model outputs a probability
            predictions.append(pred)

        # Average the prediction scores
        avg_score = np.mean(predictions)
        label = 1 if avg_score >= 0.75 else 0  # Using threshold of 0.75
        confidence = avg_score * 100 if label == 1 else (1 - avg_score) * 100
        output_text = f"{labels_dict[label]} (Average Confidence: {confidence:.2f}%)"
        print("Final Result:", output_text)

        # Display the original cropped (or full) result (converted to RGB for PIL display)
        self.processed_image = cv2.cvtColor(cropped_result, cv2.COLOR_BGR2RGB)
        display_img = Image.fromarray(self.processed_image)
        display_img = display_img.resize((700, 400), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        # Update the GUI label with the final prediction
        self.label.config(text=output_text)
        self.save_btn.config(state="normal")

    def save_image(self):
        """Save the processed image."""
        if self.processed_image is None:
            print("No processed image to save!")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", 
                                                   filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
            print(f"Image saved to {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AadhaarCropAndPredictApp(root)
    root.mainloop()
