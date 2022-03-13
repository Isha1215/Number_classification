import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam
import pickle

path="myData"
test_ratio=0.2
validation_ratio=0.2
batchSize=50
epochs_no=10
stepsPerEpoch=2000
image_shape=(32,32,3)
listdir=os.listdir(path)
no_of_classes=len(listdir)
print("Total no of classes",no_of_classes)
images=[]
classes=[]
print("Importing classes")
for x in range(0,no_of_classes):
    innerdir=os.listdir(path+"/"+str(x))
    for y in innerdir:
        img=cv2.imread(path+"/"+str(x)+"/"+y)
        img=cv2.resize(img,(image_shape[0],image_shape[1]))
        images.append(img)
        classes.append(x)
    print(x,end=" ")
print(" ")
images=np.array(images)
classes=np.array(classes)
print(images.shape)
###Splitting the data for testing training and validation
X_train,X_test,y_train,y_test=train_test_split(images,classes,test_size=test_ratio)
X_train,X_validation,y_train,y_validation=train_test_split(X_train,y_train,test_size=validation_ratio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
no_of_samples=[]
for x in range(0,no_of_classes):
    #print(x,len(np.where(y_train==x)[0]))
    no_of_samples.append(len(np.where(y_train==x)[0]))

plt.figure(figsize=(10,5))
plt.bar(range(0,no_of_classes),no_of_samples)
plt.title("No of images from each class")
plt.xlabel("Class name")
plt.ylabel("No. of images")
plt.show()

##preprocessing the images
def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #here is gets converted to a gray image so its from (0-255)
    img=cv2.equalizeHist(img) #used to equally distribute intensities throughout the picture
    img=img/255 # we want the image range between (0-1) applying normalization
    return img

X_train=np.array(list(map(preprocessing,X_train))) #maps all images of X_train into the preprocessing function and returns a list
X_test=np.array(list(map(preprocessing,X_test)))
X_validation=np.array(list(map(preprocessing,X_validation)))

####print(X_train.shape)----(6502,128,128) i.e total of 6502 rows(elements/images) 128,128 is no.of rows and cols for one image (array)

###Adding depth
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

###augmenting the images (roation,zoom,shearing) to make dataset more generic

Data_gen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,rotation_range=10,zoom_range=0.2)
Data_gen.fit(X_train)

###convert an array input into binary format eg- [1,2,3]---[[0,1,0,0],[0,0,1,0],[0,0,0,1]
"""y (input vector): A vector which has integers representing various classes in the data.
num_classes: Total number of classes. If nothing is mentioned, it considers the largest number of the input vector and adds 1, to get the number of classes.
Its default value is "None".
dtype: It is the desired data type of the output values. 
By default, it's value is 'float32'.
Output: 
This function returns a matrix of binary values (either ‘1’ or ‘0’). It has number of rows equal to the length of the input vector and number of columns equal to the number of classes."""
y_train=to_categorical(y_train,no_of_classes)
y_test=to_categorical(y_test,no_of_classes)
y_validation=to_categorical(y_validation,no_of_classes)

def Mymodel():
    no_of_filters=60
    filter_size1=(5,5)
    filter_size2=(3,3)
    pooling_size=(2,2)
    nodes=500
    model=Sequential()
    model.add((Conv2D(no_of_filters,filter_size1,input_shape=(image_shape[0],image_shape[1],1),activation='relu')))
    model.add((Conv2D(no_of_filters, filter_size1, activation='relu')))
    model.add((MaxPooling2D(pool_size=pooling_size)))
    model.add((Conv2D(no_of_filters//2, filter_size2,activation='relu')))
    model.add((Conv2D(no_of_filters // 2, filter_size2, activation='relu')))
    model.add((MaxPooling2D(pool_size=pooling_size)))
    model.add((Dropout(0.5))) ##avoid overfitting of model
    model.add((Flatten()))  ##convert 2D arrays into single long continuos vector
    model.add((Dense(nodes,activation='relu')))
    model.add((Dropout(0.5)))
    model.add((Dense(no_of_classes, activation='softmax')))
    model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=['accuracy'])
    return model


model=Mymodel()
##epoch-one travel through the entire set of images/dataset (forward and backward)
##steps=epoch_no*totalimages/batchsize
##shuffle-do we wanna shuffle data after each epoch
history=model.fit_generator(Data_gen.flow(X_train,y_train,batch_size=batchSize),epochs=epochs_no,steps_per_epoch=stepsPerEpoch,validation_data=(X_validation,y_validation),shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["training","validation"])
plt.title("loss")
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["training","validation"])
plt.title("accuracy")
plt.xlabel('epoch')
plt.show()

score=model.evaluate(X_test,y_test,verbose=0)
print("Test score= ",score[0])
print("Test accuracy= ",score[1])

model.save("model_trained.h5")

