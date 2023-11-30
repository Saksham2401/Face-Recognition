import torch
from torchvision import transforms 
from PIL import Image 
import numpy as np 
import cv2 
from mtcnn import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
detector = MTCNN()

image = cv2.imread('images/office1.jpg')

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
image_tensor = transforms.functional.to_tensor(image)



boxes = detector.detect_faces(image)

for box in boxes: 
    x, y, w, h = box['box'] 
    l_ex,l_ey = box['keypoints']['left_eye']
    r_ex,r_ey = box['keypoints']['right_eye']
    n_ex,n_ey = box['keypoints']['nose']
    l_mx,l_my = box['keypoints']['mouth_left']
    r_mx,r_my = box['keypoints']['mouth_right']

    cv2.circle(image,center=(l_ex,l_ey),color=(0,255,0),thickness=2,radius=2)
    cv2.circle(image,center=(r_ex,r_ey),color=(0,255,0),thickness=2,radius=2)
    cv2.circle(image,center=(n_ex,n_ey),color=(0,255,0),thickness=2,radius=2)
    cv2.circle(image,center=(l_mx,l_my),color=(0,255,0),thickness=2,radius=2)
    cv2.circle(image,center=(r_mx,r_my),color=(0,255,0),thickness=2,radius=2)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Faces', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()