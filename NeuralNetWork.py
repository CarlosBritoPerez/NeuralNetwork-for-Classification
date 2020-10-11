from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping
import keras
from time import time
from matplotlib import pyplot as plt 

import os
import random
import shutil
import time


# DATA SOURCE --------------------------------------------------
best_try_acc = 0
best_try = [0,0,0,0] #layers values and average
directions = ["/Manos","/Tenedores","/Trabas","/Mascarillas"]
batch_size = 60
batch_validation_size = 1

train_data_dir = '' #Directory of the photos
validation_data_dir = ''

# Short images to use as validation
for dir in directions:
  files=os.listdir(train_data_dir + dir)
  for x in range(batch_validation_size):
    d=random.choice(files)
    time.sleep(20)
    shutil.move(train_data_dir + dir + "/" + d,validation_data_dir + dir)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip = True
)

validation_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(150, 150),
        batch_size=batch_validation_size,
        class_mode='categorical')


# MODEL --------------------------------------------------

    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])


# TRAINING --------------------------------------------------

      epochs = 100

      es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15, restore_best_weights=True)

      history = model.fit_generator(
              train_generator,
              epochs=epochs,
              validation_data = validation_generator,
              callbacks = [es]

      )
      if (best_try_acc < history.history.get('accuracy')[-1]):
        best_try_acc = history.history.get('val_accuracy')[-1]
        best_try = [layer1,layer2,layer3,best_try_acc]

      plt.plot(history.history['accuracy'], label='accuracy')
      plt.plot(history.history['val_accuracy'], label='validation accuracy')

      plt.title('NeuralNet for Classification')
      plt.xlabel('Épocas')
      plt.legend(loc="lower right")

      plt.show()

# SAVING --------------------------------------------------
# Devuelve las imagenes al directorio base
list_back = []
for dir in directions:
  files=os.listdir(validation_data_dir + dir)
  for x in files:
    shutil.move(validation_data_dir + dir + "/" + x,train_data_dir + dir)

print("Best results with:")
for x in best_try:
  print(x)
model.save("")#Name of the proyect

import os
import random
import shutil
import time


directions = ["/Manos","/Tenedores","/Trabas","/Mascarillas"]

train_data_dir = '' #Directory of the photos
validation_data_dir = ''


list_back = []
for dir in directions:
  files=os.listdir(validation_data_dir + dir)
  for x in files:
    shutil.move(validation_data_dir + dir + "/" + x,train_data_dir + dir)

from google.colab import drive
drive.mount('') #In case you want to use drive
