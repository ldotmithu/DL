import os 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import glob
from tensorflow.keras.utils import to_categorical

root_dir='dataset'
dataset=[]
lables=[]

for folder in os.listdir(root_dir):
    folder_path=os.path.join(root_dir,folder)
    print(f"preprocess folders : {folder}")

    for i in os.listdir(folder_path):
        img=os.path.join(folder_path,i)
        img=load_img(img,target_size=(128,128))
        img=img_to_array(img)

        dataset.append(img)
        lables.append(folder)

dataset=np.array(dataset)/255
lables=np.array(lables)

encoder=LabelEncoder()
lables=encoder.fit_transform(lables)

X_train,X_test,y_train,y_test=train_test_split(dataset,lables,test_size=0.2,stratify=lables)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

train_gen=ImageDataGenerator(rotation_range=20,horizontal_flip=True,height_shift_range=0.2,width_shift_range=0.2,zoom_range=0.1)
train_gen.fit(X_train)

class_weight=compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)
class_weight_dict=dict(enumerate(class_weight))

y_train=to_categorical(y_train,num_classes=3)
y_test=to_categorical(y_test,num_classes=3)

base_model=tf.keras.applications.VGG19(input_shape=(128,128,3),weights='imagenet',include_top=False)

for layer in base_model.layers:
    layer.trainable=False
    
x=Flatten()(base_model.output)
x=Dense(512,activation='relu')(x)
x=Dropout(0.25)(x)
predict_layer=Dense(units=3,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=predict_layer)
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])    

#model.fit(
#train_gen.flow(X_train,y_train,batch_size=32),epochs=10,validation_data=(X_test,y_test),class_weight=class_weight_dict)

# model.evaluate(X_test,y_test)
        