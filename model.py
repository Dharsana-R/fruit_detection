import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
# loading the directories 
# importing the libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
#from keras.preprocessing import image
num_classes=3
IMAGE_SHAPE = [224, 224]  # we will keep the image size as (64,64). You can increase the size for better results. 
batch_size=32
# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = (224,224,3), weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
x = Dense(64, activation = 'relu')(x) 
x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = vgg.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
trdata = ImageDataGenerator()
train_data_gen =trdata.flow_from_directory(directory="D:\Transfer-Learning-main\Transfer-Learning-main\Train",target_size=(224,224),shuffle=False, class_mode='categorical')
tsdata = ImageDataGenerator()
test_data_gen = tsdata.flow_from_directory(directory="D:\Transfer-Learning-main\Transfer-Learning-main\Test", target_size=(224,224),shuffle=False, class_mode='categorical')


from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
epochs = 5
#checkpoint = ModelCheckpoint(filepath='finalvgg16model.h5', verbose=1, save_best_only=True)
training_steps_per_epoch = np.ceil(train_data_gen.samples / batch_size)
validation_steps_per_epoch = np.ceil(test_data_gen.samples / batch_size)
    
model.fit_generator(train_data_gen, steps_per_epoch=training_steps_per_epoch, validation_data=test_data_gen, validation_steps=validation_steps_per_epoch,
                        epochs=epochs, verbose=1)
print('Training Completed!')

Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
val_preds = np.argmax(Y_pred, axis=1)
import sklearn.metrics as metrics
val_trues =test_data_gen.classes
from sklearn.metrics import classification_report
print(classification_report(val_trues, val_preds))

Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
val_preds = np.argmax(Y_pred, axis=1)
import sklearn.metrics as metrics
val_trues =test_data_gen.classes
cm = metrics.confusion_matrix(val_trues, val_preds)
cm

keras_file="model.h5"
tf.keras.models.save_model(model,keras_file)