import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import os

# =========================================
# SETTINGS
# =========================================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset/"   # <-- your dataset folder

# =========================================
# DATA PREPROCESSING
# =========================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# =========================================
# SAVE CLASS LABELS (VERY IMPORTANT)
# =========================================
class_indices = train_data.class_indices

with open("class_names.json", "w") as f:
    json.dump(class_indices, f)

print("✅ Class labels saved!")

# =========================================
# BUILD MODEL (COMPATIBLE VERSION)
# =========================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(train_data.num_classes, activation='softmax'))

# =========================================
# COMPILE
# =========================================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================
# TRAIN
# =========================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# =========================================
# SAVE MODEL (USE .h5 FOR COMPATIBILITY)
# =========================================
model.save("plant_disease_model.h5")

print("✅ Model saved as plant_disease_model.h5")

# =========================================
# TEST WITH ONE IMAGE
# =========================================
from tensorflow.keras.preprocessing import image

sample_path = train_data.filepaths[0]

img = image.load_img(sample_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

class_index = np.argmax(prediction)
confidence = np.max(prediction)

print("Sample Prediction:", class_index)
print("Confidence:", confidence)