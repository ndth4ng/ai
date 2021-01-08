import tkinter as tk
from PIL import Image,ImageDraw
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class ImageGenerator:
    def __init__(self,parent,posx,posy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 200
        self.sizey = 200
        self.b1 = "up"
        self.xold = None
        self.yold = None
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Đoán!",width=10,bg='white',command=self.save)
        self.button.place(x=self.sizex/7,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Xóa!",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/7)+80,y=self.sizey+20)

        self.image=Image.new("RGB",(200,200),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def save(self):
        #Luu thanh hinh
        filename = "D:\Python\my_drawing.png"
        #Chuyen thanh 28x28 pixel
        self.image = self.image.resize((28, 28), 1)
        img = self.image.convert('LA')
        img.save(filename)
         
        #Chuyen hinh anh ve numpy array
        img = cv.imread('D:\Python\my_drawing.png')[:,:,0]
        img = np.invert(np.array([img]))
        #Du doan ket qua
        prediction = model.predict(img)
        #In ket qua du doan
        self.label=tk.Label(self.parent,text=str(np.argmax(prediction)))
        self.label.config(font=("Courier",140))
        self.label.place(x=self.sizex+70,y=10)
        #print(f'Ket qua: {np.argmax(prediction)}')

    def clear(self):
        self.drawing_area.delete("all")
        self.image=Image.new("RGB",(200,200),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)
        self.label.destroy()

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,width=12,fill='black')
                self.draw.line(((self.xold,self.yold),(event.x,event.y)),(0,128,0),width=12)

        self.xold = event.x
        self.yold = event.y

if __name__ == "__main__":
    #Load model da train
    model = tf.keras.models.load_model('D:\Python\digits.model')
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (500, 250, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)
    root.mainloop()

