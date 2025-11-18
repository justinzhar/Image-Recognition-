**Digit Recognition Neural Network**

A Convolutional Neural Network trained on handwritten digits with a custom UI to explore predictions.

*Overview*

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits (0–9) from 28×28 grayscale images.
The model is trained on a labeled dataset (train.csv) and evaluated using industry-standard metrics including accuracy, precision, recall, and F1-score.

A custom Tkinter GUI viewer is included, allowing users to scroll through the test.csv dataset and view each predicted digit next to its corresponding image.

*Features*

- Fully working CNN for image classification

- ~99% test accuracy

- CSV-based dataset loading (train & test)

- Automatic preprocessing (reshape + normalization)

- Full precision/recall/F1 classification report

- A clean Tkinter GUI to browse predictions

- Code structured and ready for extension (saving model, drawing pad, etc.)

- Tech Stack

Python 3.13

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Tkinter

Pillow (PIL) — for image rendering in UI
