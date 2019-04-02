import numpy
import pandas
import os
import keras
import cv2
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.utils import layer_utils
from keras.optimizers import adam
from keras.applications.resnet50 import ResNet50 #pretrained model, 50 layers
from keras.preprocessing.image import ImageDataGenerator
one = ResNet50(weights='imagenet',include_top=False, input_shape=(224, 224, 3)) #create resnet model
a = Flatten()(one.output) #add layers to output of Resnet model
a = Dense(256, activation='relu')(a)
a = Dropout(.08)(a)
a = Dense(120, activation='softmax')(a)
for layer in one.layers:
    layer.trainable = False # train only the new layers
model_finish = Model(inputs=one.input, outputs=a) # combine 
model_finish.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
train_data = ImageDataGenerator( #to change image input shape
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        )
training_set = train_data.flow_from_directory(  #create training set
        '../training_images/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
        )
test_set = train_data.flow_from_directory( #create validation set
        'validation_images/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
        )
generate=model_finish.fit_generator( #fit model
        training_set,
        steps_per_epoch=100,
        epochs=1, # of iterations
        validation_data=test_set,
        validation_steps=100
         )
number = []
for pic in os.listdir('./test/'): #create array from test images
    pic = cv2.imread('./test/'+pic)
    number.append(cv2.resize(pic,(224, 224)))
number = numpy.array(number, numpy.float32) #convert to numpy array
final_prod = model_finish.predict(number) #predict test set
column_names = training_set.class_indices #columns for csv
final = pandas.DataFrame(final_prod) #create final csv with output of predicted test set
train = pandas.read_csv('./labels.csv')#read csv to use columns
test = pandas.read_csv('./sample_submission.csv') #use sample submission csv for table layout
col = pandas.Series(train['breed']) #use columns from labels.csv to append to the headers of final csv
one_hot = pandas.get_dummies(col, sparse = True)
final.columns = one_hot.columns.values #append columns from labels.csv
final.insert(0, 'id', test['id']) #append id values from sample csv
final.to_csv('./final.csv',sep=",") #create submission file
