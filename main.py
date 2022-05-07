from threading import Thread
from time import sleep

from traffic_sign_recognition import recognize
from tkinter import *
from tkinter.ttk import *
import tkinter
import PIL.Image
import PIL.ImageTk
import cv2

# GUI
windows = Tk()
windows.title("Traffic Sign Recognition")
windows.geometry("400x700")

lblDetect = tkinter.Label(windows, text="Camera", font=("Roboto", 16))
lblDetect.pack()

video = cv2.VideoCapture(0)
canvas_w = video.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
canvas_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2
camera_canvas = Canvas(windows, width=canvas_w, height=canvas_h, bg="grey")
camera_canvas.pack()

image_canvas = Canvas(windows, width=canvas_w, height=canvas_h, bg="black")
image_canvas.pack()

photo = None
count = 0


def update_detect():
    sleep(10)
    img, _ = recognize.detect(photo, classes, net)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    image_canvas.create_image(0, 0, image=img, anchor=tkinter.NW)


def update_frame():
    global camera_canvas, photo, count
    ret, frame = video.read()
    photo = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(photo))
    camera_canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

    count = count + 1
    if count % 10 == 0:
        thread = Thread(target=update_detect)
        thread.start()

    windows.after(15, update_frame)


button = Button(windows, text="Start", command=update_frame)
button.pack()

ta = Text(windows, width=40, height=8)
ta.pack()

windows.mainloop()

if __name__ == '__main__':
    # Load class names.
    classesFile = r"data\models\traffic_sign.names"
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the weight files to the model and load the network using them.
    modelWeights = r"data\models\traffic_sign.onnx"
    net = cv2.dnn.readNet(modelWeights)


