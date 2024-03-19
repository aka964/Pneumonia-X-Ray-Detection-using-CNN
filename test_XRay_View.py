import cv2
import numpy as np
import matplotlib.pyplot as plt

#View Creation Libraries
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

#deepLearning Libraries
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D

img_dim = 256

#input layer
inputs = Input(shape=(img_dim, img_dim, 3))

#First conv block(hidden layer)
model = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
model = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(model)
model = MaxPool2D(pool_size=(2, 2))(model)

#Second conv block(hidden layer)
model = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(model)
model = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(model)
model = BatchNormalization()(model)
model = MaxPool2D(pool_size=(2, 2))(model)

#Third conv block(hidden layer)
model = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(model)
model = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(model)
model = BatchNormalization()(model)
model = MaxPool2D(pool_size=(2, 2))(model)

#Fourth conv block(hidden layer)
model = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(model)
model = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(model)
model = BatchNormalization()(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(rate= 0.2)(model)

#Fully Connected layer
model = Flatten()(model)
model = Dense(units=512, activation='relu')(model)
model = Dropout(rate=0.7)(model)
model = Dense(units=128, activation='relu')(model)
model = Dropout(rate=0.5)(model)
model = Dense(units=64, activation='relu')(model)
model = Dropout(rate=0.3)(model)

#Output layer
output = Dense(units=1, activation='sigmoid')(model)

#Creating model and compiling
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def preprocess_image(image_path, img_dim):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_dim, img_dim))
    img = img.astype('float32') / 255
    return img

model.load_weights('efficient_weight.hdf5')

def predict_image():
    image_path = image_path_entry.get()
    test_image = preprocess_image(image_path, img_dim)
    
    prediction = model.predict(np.expand_dims(test_image, axis=0))
    
    img = Image.fromarray((test_image * 255).astype(np.uint8))
    img = img.resize((img_dim, img_dim))
    img_tk = ImageTk.PhotoImage(image=img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk
    
    prediction_label.configure(text=f"{'Normal' if prediction[0][0] < 0.5 else 'Pneumonia'}")

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    image_path_entry.delete(0, tk.END)
    image_path_entry.insert(0, file_path)

# Create the main application window
root = tk.Tk()
root.title("Chest X-ray Checker")

image_path_entry = ttk.Entry(root, width=50)
image_path_entry.grid(row=0, column=0, padx=10, pady=10)

browse_button = ttk.Button(root, text="Browse", command=browse_image)
browse_button.grid(row=0, column=1, padx=10, pady=10)

predict_button = ttk.Button(root, text="Check", command=predict_image)
predict_button.grid(row=0, column=2, padx=10, pady=10)

image_label = ttk.Label(root)
image_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

prediction_label = ttk.Label(root, text="")
prediction_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()