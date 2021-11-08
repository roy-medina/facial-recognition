# this file is used to detect face 
# and then store the data of the face 
import cv2 
import numpy as np


# import the file where data is 
# stored in a csv file format 
import npwriter
name = input("Enter your name: ")


# this is used to access the web-cam 
# in order to capture frames 
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_720p()



# this is class used to detect the faces as provided 
# with a haarcascade_frontalface_default.xml file as data 
f_list = []

while True: 
    ret, frame = cap.read() 

    # converting the image into gray 
    # scale as it is easy for detection 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # detect multiscale, detects the face and its coordinates 
    faces = classifier.detectMultiScale(gray, 1.5, 5) 

    # this is used to detect the face which 
    # is closest to the web-cam on the first position 
    faces = sorted(faces, key = lambda x: x[2]*x[3], 
            reverse = True)
    # only the first detected face is used 
    faces = faces[:1] 

    # len(faces) is the number of python recog.py

    # faces showing in a frame 
    if len(faces) == 1: 
    # this is removing from tuple format  
        face = faces[0] 

        # storing the coordinates of the 
        # face in different variables 
        x, y, w, h = face
        # this is will show the face 
        # that is being detected  
        im_face = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), 
            (255, 0, 0), 3)

        cv2.imshow("face", im_face)

        # Run this code if you want to always save face data
        # gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY) 
        # gray_face = cv2.resize(gray_face, (100, 100)) 
        # print(len(f_list), type(gray_face), gray_face.shape)
        # # this will append the face's coordinates in f_list 
        # f_list.append(gray_face.reshape(-1)) 
    
        if not ret: 
            continue

    else: 
        print("face not found")


    cv2.imshow("full", frame)
    key = cv2.waitKey(1)

    # Run this code if you want to always save face data
    # npwriter.write(name, np.array(f_list))


    # this will break the execution of the program 
    # on pressing 'q' and will click the frame on pressing 'c' 
    if key & 0xFF == ord('q'): 
        break

    # Run this code if you want to manually choose when to save face data
    elif key & 0xFF == ord('c'): 
        if len(faces) == 1: 
            gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY) 
            gray_face = cv2.resize(gray_face, (100, 100)) 
            print(len(f_list), type(gray_face), gray_face.shape)
            # this will append the face's coordinates in f_list 
            f_list.append(gray_face.reshape(-1)) 
        else: 
            print("face not found")
        # this will store the data for detected 
        # face 10 times in order to increase accuracy 
        if len(f_list) == 10: 
        
            # declared in npwriter 
            npwriter.write(name, np.array(f_list))



cap.release() 
cv2.destroyAllWindows()