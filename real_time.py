# import torch
# from torchvision import transforms 
# from PIL import Image 
# import numpy as np 
import cv2 
from mtcnn import MTCNN

cap = cv2.VideoCapture(0)
detector = MTCNN()


while True:

    ret,frame = cap.read()

    output = detector.detect_faces(frame)

    for single_output in output:
        x,y,w,h = single_output['box']
        cv2.rectangle(frame,pt1=(x,y), pt2=(x+w,y+h),color=(255,0,0),thickness=2)
    cv2.imshow('win',frame)

    if cv2.waitKey(1) & 0xFF == ('x'):
        break

cv2.destroyAllWindows()