from tkinter import *
import os
import cv2
import time
import uuid

IMAGE_PATH = "Image"

# membuat window
window = Tk()
window.title("Collecting Images for Machine Learning")

# membuat label dan input box untuk jumlah label
num_of_labels_label = Label(window, text="Number of Labels:")
num_of_labels_label.grid(row=0, column=0)
num_of_labels_entry = Entry(window)
num_of_labels_entry.grid(row=0, column=1)

# membuat tombol untuk menambahkan nama-nama label
add_label_button = Button(window, text="Add Label", command=lambda: add_label_entry())
add_label_button.grid(row=1, column=1)

# fungsi untuk menambahkan kotak input label
def add_label_entry():
    num_of_labels = int(num_of_labels_entry.get())
    for i in range(num_of_labels):
        label_name_entry = Entry(window)
        label_name_entry.grid(row=i+2, column=1)
        label_names.append(label_name_entry)

# membuat kotak/box untuk nama-nama label
label_names = []

# membuat label dan input box untuk jumlah gambar per label
number_of_images_label = Label(window, text="Number of Images per Label:")
number_of_images_label.grid(row=100, column=0) # diatur besar agar tidak tumpang tindih dengan kotak input
number_of_images_entry = Entry(window)
number_of_images_entry.grid(row=100, column=1) # diatur besar agar tidak tumpang tindih dengan kotak input

def collect_images():
    # mengambil nilai input dari pengguna
    num_of_labels = int(num_of_labels_entry.get())
    labels = [label_names[i].get() for i in range(num_of_labels)]
    number_of_images = int(number_of_images_entry.get())

    for label in labels:
        img_path = os.path.join(IMAGE_PATH, label)
        os.makedirs(img_path)
        cap = cv2.VideoCapture(0)
        print("Collecting images for {}".format(label))
        time.sleep(5)
        for imgnum in range(number_of_images):
            ret, frame = cap.read()
            imagename = os.path.join(IMAGE_PATH, label, label + '.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(imagename, frame)
            cv2.imshow('frame', frame)
            time.sleep(2)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        cap.release()

# membuat tombol untuk memulai pengambilan gambar
start_button = Button(window, text="Start", command=collect_images)
start_button.grid(row=200, column=1) # diatur besar agar tidak tumpang tindih dengan kotak input

# menjalankan window
window.mainloop()
