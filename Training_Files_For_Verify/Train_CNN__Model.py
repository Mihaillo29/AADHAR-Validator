# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Add
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# from sklearn.model_selection import train_test_split
# from sklearn.utils import class_weight
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score  # <-- New imports

# # Load Data
# data = np.load('data.npy')
# target = np.load('target.npy')

# # Split data for training and testing
# train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=42)

# # Compute class weights to handle imbalance
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_target),
#     y=train_target
# )
# class_weights_dict = dict(enumerate(class_weights))
# print(f"Class weights: {class_weights_dict}")

# # Data Augmentation
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Residual Block Function
# def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
#     y = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
#     y = Activation('relu')(y)
#     y = Conv2D(filters, kernel_size, padding='same')(y)
    
#     # Adjust input dimensions if needed (for addition)
#     if x.shape[-1] != filters or strides != (1, 1):
#         x = Conv2D(filters, (1, 1), strides=strides, padding='same')(x)
    
#     out = Add()([x, y])  # Residual connection
#     out = Activation('relu')(out)
#     return out

# # Generate Model
# inputs = Input(shape=data.shape[1:])
# x = Conv2D(32, (3, 3), padding='same')(inputs)
# x = Activation('relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)

# # Add residual blocks
# x = residual_block(x, 32)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)

# x = residual_block(x, 64)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)

# x = Flatten()(x)
# x = Dropout(0.5)(x)
# x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
# outputs = Dense(1, activation='sigmoid')(x)

# model = Model(inputs, outputs)

# # Compile with binary crossentropy
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Early Stopping Callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train our model with class weights
# history = model.fit(
#     datagen.flow(train_data, train_target, batch_size=32),
#     epochs=43,
#     validation_data=(test_data, test_target),
#     class_weight=class_weights_dict,
#     callbacks=[early_stopping]
# )

# # Save trained model
# model.save("Card_Detector_ResNet.keras")

# # Evaluate on test set
# test_loss, test_accuracy = model.evaluate(test_data, test_target)
# print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# # --- Additional Metrics ---

# # Get predictions on the test set (threshold at 0.5)
# predictions = model.predict(test_data)
# predicted_labels = (predictions >= 0.5).astype(int).flatten()

# print("Classification Report:")
# print(classification_report(test_target, predicted_labels))
# print("Confusion Matrix:")
# print(confusion_matrix(test_target, predicted_labels))
# print("F1 Score:", f1_score(test_target, predicted_labels))
# print("Recall Score:", recall_score(test_target, predicted_labels))
# print("Precision Score:", precision_score(test_target, predicted_labels))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score  # <-- New imports

# Load Data
data = np.load('data.npy')
target = np.load('target.npy')

# Split data for training and testing
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=42)

# Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_target),
    y=train_target
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weights_dict}")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,    # <-- Added vertical flip augmentation
    fill_mode='nearest'
)

# Residual Block Function
def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    y = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size, padding='same')(y)
    
    # Adjust input dimensions if needed (for addition)
    if x.shape[-1] != filters or strides != (1, 1):
        x = Conv2D(filters, (1, 1), strides=strides, padding='same')(x)
    
    out = Add()([x, y])  # Residual connection
    out = Activation('relu')(out)
    return out

# Generate Model
inputs = Input(shape=data.shape[1:])
x = Conv2D(32, (3, 3), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Add residual blocks
x = residual_block(x, 32)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = residual_block(x, 64)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compile with binary crossentropy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train our model with class weights and data augmentation applied on the fly
history = model.fit(
    datagen.flow(train_data, train_target, batch_size=32),
    epochs=60,
    validation_data=(test_data, test_target),
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

# Save trained model
model.save("Card_Detector_ResNet.keras")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_data, test_target)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# --- Additional Metrics ---

# Get predictions on the test set (threshold at 0.5)
predictions = model.predict(test_data)
predicted_labels = (predictions >= 0.5).astype(int).flatten()

print("Classification Report:")
print(classification_report(test_target, predicted_labels))
print("Confusion Matrix:")
print(confusion_matrix(test_target, predicted_labels))
print("F1 Score:", f1_score(test_target, predicted_labels))
print("Recall Score:", recall_score(test_target, predicted_labels))
print("Precision Score:", precision_score(test_target, predicted_labels))
