import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/soybean_rust_mobilenet.keras"
IMG_SIZE = 224

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found!")
    print("Expected location:")
    print(MODEL_PATH)
    sys.exit()


print("\nLoading Soybean Rust Detection Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# GET IMAGE PATH

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = input("\nEnter the path of the soybean leaf image: ").strip()


# Remove quotes if user copies path with quotes
image_path = image_path.strip('"').strip("'")

# CHECK IMAGE

if not os.path.exists(image_path):
    print("\nERROR: Image file not found!")
    print("Check the image path and try again.")
    sys.exit()


# ==========================================
# LOAD IMAGE
# ==========================================

print("\nAnalyzing image...")

img = image.load_img(
    image_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)


# ==========================================
# CONVERT IMAGE TO ARRAY
# ==========================================

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)


# ==========================================
# MOBILENETV2 PREPROCESSING
# ==========================================

img_array = preprocess_input(img_array)


# ==========================================
# MAKE PREDICTION
# ==========================================

prediction = model.predict(img_array, verbose=0)

probability = float(prediction[0][0])


# ==========================================
# DETERMINE CLASS
# ==========================================

if probability >= 0.5:

    predicted_class = "Soybean Rust"
    confidence = probability * 100

else:

    predicted_class = "Healthy Soybean Leaf"
    confidence = (1 - probability) * 100


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n========================================")
print("     SOYBEAN DISEASE DETECTION")
print("========================================")

print("\nImage:")
print(image_path)

print("\nPrediction:")
print(predicted_class)

print(f"\nConfidence: {confidence:.2f}%")

print("\n========================================")

if predicted_class == "Soybean Rust":

    print("⚠️ RESULT: SOYBEAN RUST DETECTED")
    print("The leaf shows characteristics associated")
    print("with Soybean Rust.")

else:

    print("✅ RESULT: HEALTHY SOYBEAN LEAF")
    print("No Soybean Rust was detected by the model.")

print("========================================\n")
