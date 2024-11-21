import cv2
import os
import cv2.data
import numpy as np

from keras.src.saving.saving_api import load_model
from keras._tf_keras.keras.models import Model

test_case = 2# Choose test case, we have two test case 

if os.path.exists('khuonmat.h5'):
    Model = load_model('khuonmat.h5')
else:
    print('File not found. Please check the path.')

filename = os.path.join('data_test',str(test_case)+'.jpg')
image = cv2.imread(filename)
if image is None:
    print("Can't read image.Please check your path.")
    exit()    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fontface = cv2.FONT_HERSHEY_SIMPLEX
faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    roi_gray = img_gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (64, 64))
    roi_gray = roi_gray.reshape((1, 64, 64, 1))

    result = Model.predict(roi_gray)
    final = np.argmax(result)
# Change the display name to match the key
####################################
    names = {0: "bao_lam",
            1: "q_hai", 
            2: "tran_thanh", 
            3: "truong_giang", 
            4: "tu_long", 
            5: "xuan_bac"
            }
#####################################    
    if final in names:
        cv2.putText(image, names[final], (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)

cv2.imshow('Training', image)
cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
cv2.waitKey(0)
cv2.destroyAllWindows()