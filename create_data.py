import cv2
import numpy as np
import cv2.data
import os
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

data_path = 'data'

def getFaces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc hình ảnh. Kiểm tra đường dẫn tệp.")
        exit(3)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(image_gray, 1.3,5)
    count = 0  
    for (x,y,w,h) in faces:
        image_faces = cv2.resize(image[y+2:y+h-3,x+2:x+w-3],(64,64))
        cv2.imwrite(image_path.replace('data','train_data'),image_faces)
        count+=1              
           
for whatelse in os.listdir(data_path): 
    data_train = os.path.join(data_path,whatelse) 
    for sub_whatelse in os.listdir(data_train):
        img_path = os.path.join(data_path,whatelse,sub_whatelse)
        if not os.path.isdir(data_train.replace('data','train_data')):
            os.makedirs(data_train.replace('data','train_data'))
        if img_path.endswith('jpg'):
            getFaces(img_path)
cv2.destroyAllWindows()