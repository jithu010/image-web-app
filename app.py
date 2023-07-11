

import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow
import  tensorflow as tf 


import warnings
warnings.filterwarnings('ignore')   
import os 

st.title('Potato Plant Identification')
#setting constants 
Batch_size=32
Image_size=256,
Epochs=6
Channels=3

#loading images from file directory 

dataset=tf.keras.preprocessing.image_dataset_from_directory(r'C:\Users\jithu\OneDrive\Pictures\Desktop\PlantVillage',batch_size=32,
                                                            image_size=(256,256), seed=666, shuffle=True)

classes=dataset.class_names
#st.write(classes)
#for image_batch ,label_batch in dataset.take(1):
 #   st.write(image_batch.shape)
  #  st.write(label_batch.numpy())



#for image_batch,label_batch in dataset.take(1):
 #   for j in range(12):
  #      ax=plt.subplot(3,4,j+1)
   #     st.write(classes[label_batch[j]])
    #    st.image(image_batch[j].numpy().astype('uint8'))



length_ds=len(dataset)
training_ds_size=0.8
test_ds_size=0.1
val_ds_size=0.1


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
#st.write(len(train_ds),len(test_ds),len(val_ds))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Resizing(256,256),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2),
  tf.keras.layers.experimental.preprocessing.RandomHeight(factor=0.2),
  tf.keras.layers.experimental.preprocessing.RandomWidth(factor=0.2),
  tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .2)
])

train_image_data=train_ds.map(
    lambda x,y: (data_augmentation(x,training=True),y)).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape=(32,256,256,3)


st.text("Shape of training dataset after data augmentation")
st.write(len(train_image_data))


model = tf.keras.models.Sequential([
    resize_and_rescale,
    tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
]) 


model.build(input_shape=input_shape)
print(model.summary())

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


@st.cache_resource
def history_of_model(_model):
    history = model.fit(
    train_image_data,
    batch_size=32,
    validation_data=val_ds,
    verbose=1,
    epochs=20) 
    return history

history_of_model(model)


scores = model.evaluate(test_ds) 
st.write(scores)






image_upload=st.file_uploader(label="upload image",)
if image_upload is not None:
    image_load=tf.keras.preprocessing.image.load_img(image_upload)
    image_array=tf.keras.utils.img_to_array(image_load)
    image_resize=tf.image.resize(image_array,(256,256))
    img_array = tf.expand_dims(image_resize, 0) 
    predictions=model.predict(img_array)
    
    st.write("The Uploaded Image Belongs to ") 
    st.header(classes[np.argmax(predictions[0])])
    st.success(predictions[0])











