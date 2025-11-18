import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk

# Change this to your actual file path
df = pd.read_csv(r"C:\Users\justin\Downloads\coding\ImageRecog\train.csv\train.csv")

# Separate features (X) and labels (y)
y = df["label"].values
X = df.drop("label", axis=1).values   # shape (n, 784)

# ------------------------------------------------------
# 3. Reshape to 28x28x1 and normalize
# ------------------------------------------------------
X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# ------------------------------------------------------
# 4. Show 1 sample image from dataset
# ------------------------------------------------------
plt.imshow(X[0].reshape(28, 28), cmap="gray")
plt.title(f"Label: {y[0]}")
plt.show()

# ------------------------------------------------------
# 5. Train-test split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------
# 6. Build CNN model
# ------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ------------------------------------------------------
# 7. Train the model
# ------------------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

# ------------------------------------------------------
# 8. Evaluate
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# ------------------------------------------------------
# 9. Precision / Recall / F1
# ------------------------------------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Load final test data
test_df = pd.read_csv(r"C:\Users\justin\Downloads\coding\ImageRecog\test.csv\test.csv")
X_test_ui = test_df.values.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Preprocess
pred_probs_ui = model.predict(X_test_ui)
predictions_ui = np.argmax(pred_probs_ui, axis=1)

num_images = X_test_ui.shape[0]

# ------------------------------------------------------
# Build a simple Tkinter UI to browse images + labels
# ------------------------------------------------------
current_index = 0  # start at first image

root = tk.Tk()
root.title("Digit Viewer - Test Data with Predicted Labels")

# Image label
image_label = tk.Label(root)
image_label.pack(pady=10)

# Text label (for predicted label + index)
text_label = tk.Label(root, font=("Arial", 16))
text_label.pack(pady=5)

def update_view():
    """Update the image and text for the current_index."""
    global photo  # keep a reference so image doesn't get garbage collected

    # Get the image data (28x28) and scale up for better visibility
    img_array = (X_test_ui[current_index].reshape(28, 28) * 255).astype("uint8")
    img = Image.fromarray(img_array, mode="L")      # "L" = grayscale
    img = img.resize((280, 280), Image.NEAREST)     # scale up 10x

    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)

    pred_label = predictions_ui[current_index]
    text_label.config(
        text=f"Image {current_index + 1} / {num_images}  |  Predicted Label: {pred_label}"
    )

def next_image():
    global current_index
    if current_index < num_images - 1:
        current_index += 1
        update_view()

def prev_image():
    global current_index
    if current_index > 0:
        current_index -= 1
        update_view()

# Buttons to navigate
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

prev_btn = tk.Button(btn_frame, text="Previous", command=prev_image, width=10)
prev_btn.pack(side="left", padx=5)

next_btn = tk.Button(btn_frame, text="Next", command=next_image, width=10)
next_btn.pack(side="left", padx=5)

# Initialize the first view
update_view()

# Start the UI loop
root.mainloop()