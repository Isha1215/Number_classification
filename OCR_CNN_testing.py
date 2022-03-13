import numpy as np
import cv2
from keras.models import load_model
import pickle
import os

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
print()
def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #here is gets converted to a gray image so its from (0-255)
    img=cv2.equalizeHist(img) #used to equally distribute intensities throughout the picture
    img=img/255 # we want the image range between (0-1) applying normalization
    return img
model=load_model("C:/Users/ishad/Documents/prog/PycharmProjects/Number_classification/model_trained.h5")
print(model.summary())
while True:
    success,img_orignal=cap.read()
    img=np.asarray(img_orignal)
    img=cv2.resize(img,(32,32))
    img=preprocessing(img)
    img_orignal=cv2.resize(img_orignal,(0,0),None,0.3,0.3)
    cv2.imshow("pre-processed image",img)
    img=img.reshape(1,32,32,1)
    class_index=int(model.predict_classes(img))
    predictions=model.predict(img)
    probVal=np.amax(predictions)
    if probVal>0.8:
        cv2.putText(img_orignal,str(class_index),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
    cv2.imshow("orignal image", img_orignal)
    if cv2.waitKey(1) & 0XFF==ord('p'):
        break


"""list_dir=os.listdir("C:/Users/Dell/PycharmProjects/Number_classification/A")
print(list_dir)
for i in list_dir:
    img_orignal=cv2.imread("C:/Users/Dell/PycharmProjects/Number_classification/A"+"/"+i)
    img=np.asarray(img_orignal)
    img=cv2.resize(img,(32,32))
    img=preprocessing(img)
    img=img.reshape(1,32,32,1)
    classIndex=(int(model.predict_classes(img)))
    predictions=model.predict(img)
    probVal=np.amax(predictions)
    print(classIndex,probVal)"""