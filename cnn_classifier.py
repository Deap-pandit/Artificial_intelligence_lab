# ==============================
# Almond vs Peanut CNN Classifier
# ==============================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os, time, random
from tensorflow.keras.preprocessing import image

# ==============================
# 1️⃣ Dataset Directories
# ==============================
train_dir = 'data/train'  # contains 'almond/' and 'peanut/' subfolders
test_dir = 'data/test'

# ==============================
# 2️⃣ Data Preprocessing & Augmentation
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=16,
    subset='training',
    class_mode='categorical'
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=16,
    subset='validation',
    class_mode='categorical'
)

# ==============================
# 3️⃣ CNN Model Definition
# ==============================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==============================
# 4️⃣ Training the Model
# ==============================
start_time = time.time()
history = model.fit(train_gen, epochs=15, validation_data=val_gen)
train_time = time.time() - start_time
print(f"\n✅ Training completed in {train_time:.2f} seconds.\n")

# ==============================
# 5️⃣ Testing the Model
# ==============================
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

start_test = time.time()
test_loss, test_acc = model.evaluate(test_gen)
test_time = time.time() - start_test

print(f"\n  Testing time: {test_time:.2f} seconds")
print(f"  Test Accuracy: {test_acc*100:.2f}%\n")

# ==============================
# 6️⃣ Plot Accuracy & Loss Curves
# ==============================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Epoch vs Accuracy')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Epoch vs Loss')
plt.show()

# ==============================
# 7️⃣ Predict 10 Random Test Images
# ==============================
class_labels = list(test_gen.class_indices.keys())

# Collect all image file paths from test directory
all_test_images = []
for class_name in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_name)
    if os.path.isdir(class_path):
        all_test_images += [os.path.join(class_path, fname) for fname in os.listdir(class_path)]

# Pick 10 random images
sample_images = random.sample(all_test_images, 10)

plt.figure(figsize=(15, 8))
correct = 0

for i, img_path in enumerate(sample_images):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred = model.predict(img_array, verbose=0)
    predicted_class = class_labels[np.argmax(pred)]
    confidence = np.max(pred)
    
    # Get actual class
    actual_class = os.path.basename(os.path.dirname(img_path))
    if predicted_class == actual_class:
        correct += 1
    
    # Display
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"Pred: {predicted_class}\nAct: {actual_class}\nConf: {confidence:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"\n  Correct Predictions: {correct}/10 ({correct*10}%)")

# ==============================
# 8️⃣ Save Model
# ==============================
model.save("almond_peanut_cnn_model.h5")
print("\n  Model saved as 'almond_peanut_cnn_model.h5'")