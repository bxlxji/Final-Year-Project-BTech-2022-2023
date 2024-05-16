import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import pickle
import numpy as np
import os
import re


class GUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.res = ""
        self.setting_gui()
        path = 'models/model-kk-15p-ta81_29.pickle'
        current_directory = os.path.dirname(__file__)
        parent_directory = current_directory
        for i in range(0, 1):
            parent_directory = os.path.split(parent_directory)[0]
        file_path = os.path.join(parent_directory, path)
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        self.classes = [
            'almond',
            'hibiscus',
            'money plant',
            'rose',
            'guava',
            'jasmine',
            'lemon',
            'mango',
            'sapota (naseberry)',
            'curry leaf',
            'custard apple',
            'miracle',
            'sacred fig',
            'pheel wheel',
            'neem'
        ]

    def setting_gui(self):
        self.file_path = tk.StringVar()
        self.file_path.set("")

        self.label = tk.Label(self, text="Upload an image:")
        self.label.pack(padx=10, pady=10)

        self.filename_entry = tk.Entry(
            self, textvariable=self.file_path, width=40)
        self.filename_entry.pack(padx=8, pady=8)

        self.browse_button = tk.Button(
            self, text="Browse", command=self.browse_file)
        self.browse_button.pack(padx=10, pady=5)

        self.upload_button = tk.Button(
            self, text="Upload", command=self.upload_file)
        self.upload_button.pack(padx=10, pady=5)

        self.image_label = tk.Label(self)
        self.image_label.pack()

        self.label_2 = tk.Label(self, text="Predicted label class")
        self.label_2.pack(padx=10, pady=5)

        self.label_2 = tk.Label(self, text="[class]")
        self.label_2.pack(padx=10, pady=10)

        self.style = ttk.Style(self)
        self.style.configure('Label', font=('Helvetica', 12))
        self.style.configure('Button', font=('Helvetica', 11))

    def browse_file(self):
        self.label_2.config(text="[class]")
        file_path = filedialog.askopenfilename()
        self.file_path.set(file_path)
        self.load_image(file_path)

    def load_image(self, file_path):
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((320, 240))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

    def upload_file(self):
        file_path = self.file_path.get()
        # checking for valid file path
        x = re.findall(".jpg", file_path)
        if file_path == '' or len(x) == 0:
            self.label_2.config(
                text="Please select a file/Please provide a .jpg file")
            return
        print("Uploading file:", file_path)
        target = cv2.imread(file_path)
        target = cv2.resize(target, (50, 50))
        X = np.array([target])
        X = X / 255
        result = self.model.predict(X)
        self.res = self.classes[np.argmax(result)]
        print(self.res)
        self.label_2.config(text=self.res)


window = tk.Tk()
window.title("Plant Leaf Identification using CNN")

window.geometry("500x500")
gui = GUI(master=window)
gui.mainloop()
