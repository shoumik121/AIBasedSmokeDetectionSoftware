import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import time
import glob
import pandas as pd
import tkinter as tk
from tkinter import filedialog

file_path = None

if file_path!= "":

    print("Install dropbox and select the folder your teacher shared with you and make sure your dropbox is turned on while running this program.")

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askdirectory()
    print(file_path)
    
    id_number = input("Enter your full id: ")


previous_time="X"  


#import socket


#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        t = time.localtime()
        
        current_time = time.strftime("%H:%M:%S", t)
              

        print(current_time,predicted_emotion)
        
        #dir = r,file_path
        file = open(os.path.join(file_path, str(id_number) +'.csv'), "a")
        


        
        if os.stat(file_path+"/"+str(id_number)+'.csv').st_size == 0:
            file.write( "Time" + "," + str(id_number) + "\n")
        else:
          
            if (current_time==previous_time):
                pass
            else:
            
                #file.write( "TimeX" + "," + "EmotionY" + "\n")
                if (predicted_emotion=="happy"):
                    file.write( current_time + "," + "7" + "\n")
                elif (predicted_emotion=="neutral"):
                    file.write( current_time + "," + "5" + "\n")
                elif (predicted_emotion=="surprise"):
                    file.write( current_time + "," + "6" + "\n")
                elif (predicted_emotion=="fear"):
                    file.write( current_time + "," + "2" + "\n")
                elif (predicted_emotion=="angry"):
                    file.write( current_time + "," + "3" + "\n")
                elif (predicted_emotion=="sad"):
                    file.write( current_time + "," + "1" + "\n")
                elif (predicted_emotion=="disgust"):
                    file.write( current_time + "," + "4" + "\n")            
        
        previous_time=current_time
        
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows