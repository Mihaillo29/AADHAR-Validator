import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load models
@st.cache_resource
def load_detection_models():
    yolo_model = YOLO("best.pt")  # YOLOv8 model for cropping/detection
    tf_model = load_model("Card_Detector_ResNet.keras")  # TensorFlow model for classification
    return yolo_model, tf_model

# Labels for binary classification
labels_dict = {1: 'Aadhar Card', 0: 'Not an Aadhar Card'}

def adjust_orientation(img):
    """Check and rotate image to horizontal if vertical."""
    height, width = img.shape[:2]
    if height > width:  # Vertical image
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def preprocess_for_model(image):
    """Convert image to grayscale, resize, normalize, and reshape for model input."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_image, (100, 100))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 100, 100, 1)
    return reshaped

def process_image(image, yolo_model, tf_model):
    """
    Process the uploaded image for Aadhaar card detection.
    
    Args:
        image (numpy.ndarray): Input image
        yolo_model (YOLO): Trained YOLO model for detection
        tf_model (keras.Model): Trained classification model
    
    Returns:
        tuple: Processed image, label, confidence
    """
    # Adjust orientation to horizontal
    img = adjust_orientation(image)

    # Step 1: Detect regions with YOLOv8
    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Format: [x_min, y_min, x_max, y_max]

    # If no region is detected, use the entire image for verification
    if len(boxes) == 0:
        st.warning("No regions detected! Using the full image for verification.")
        cropped_result = img
    else:
        # Determine the bounding box that encloses all detected regions
        x_min = int(np.min(boxes[:, 0]))
        y_min = int(np.min(boxes[:, 1]))
        x_max = int(np.max(boxes[:, 2]))
        y_max = int(np.max(boxes[:, 3]))
        cropped_result = img[y_min:y_max, x_min:x_max]
        st.info(f"Detected region cropped: ({x_min}, {y_min}) to ({x_max}, {y_max})")

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
        processed = preprocess_for_model(im)
        pred = tf_model.predict(processed)[0][0]  # Assuming model outputs a probability
        predictions.append(pred)

    # Average the prediction scores
    avg_score = np.mean(predictions)
    label = 1 if avg_score >= 0.75 else 0  # Using threshold of 0.75
    confidence = avg_score * 100 if label == 1 else (1 - avg_score) * 100

    return cropped_result, labels_dict[label], confidence

def main():
    st.title("Aadhaar Card Detection App")
    
    # Load models
    yolo_model, tf_model = load_detection_models()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, channels="BGR")
        
        # Process image
        processed_image, label, confidence = process_image(image, yolo_model, tf_model)
        
        # Display processed image
        st.subheader("Processed Image")
        st.image(processed_image, channels="BGR")
        
        # Display results
        st.subheader("Results")
        st.write(f"**Classification:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        # Option to download processed image
        processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        with st.expander("Download Processed Image"):
            buffered = st.download_button(
                label="Download image",
                data=processed_pil_image.tobytes(),
                file_name="processed_image.jpg",
                mime="image/jpeg"
            )

if __name__ == "__main__":
    main()