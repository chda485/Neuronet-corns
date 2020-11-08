import os, shutil
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
#from keras.applications import VGG16
from keras.models import load_model 
import numpy as np
#!!!!!!!!!НЕОБХОДИМО ДЛЯ ВИЗУАЛИЗАЦИИ ФИЛЬТРОВ СЕТЕЙ
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

train = '/home/dmitry/progi/corns/train/'
validation = '/home/dmitry/progi/corns/validation/'
test = '/home/dmitry/progi/corns/test/'
Train_need = False
Proba = '/home/dmitry/progi/corns/train/proba.jpg'

#conv_base = VGG16(weights='imagenet', include_top = False, input_shape=(150,150,3))

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3),activation = 'relu', input_shape = (150,150,3)))
    model.add(layers.Conv2D(32, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3),activation = 'relu'))
    model.add(layers.Conv2D(64, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3),activation = 'relu'))
    model.add(layers.Conv2D(128, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3),activation = 'relu'))
    model.add(layers.Conv2D(128, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model
    
def train_model(model, train_dir, validation_dir):   
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size = (150, 150),
        batch_size = 5, class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir, target_size = (150, 150),
        batch_size = 2, class_mode='categorical')     
    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizers.RMSprop(lr = 1e-4),
                  metrics=['acc'])
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=16, epochs=100,
        validation_data=validation_generator,
        validation_steps=5)
    model.save('corns.h5')
    return history

def show(history):
    loss = history.history['loss']
    vall_loss = history.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'bo', label = 'Training loss')
    plt.plot(epochs, vall_loss, 'b', label='Validation loss')
    plt.title('Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    acc = history.history['acc']
    vall_acc = history.history['val_acc']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, acc, 'bo', label = 'Training acc')
    plt.plot(epochs, vall_acc, 'b', label='Validation acc')
    plt.title('Training')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()

if Train_need:
    model = build_model()
   #print(model.summary())
    hist = train_model(model, train, validation)
    show(hist)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            test, target_size = (150, 150),
            batch_size = 2, class_mode='categorical') 
    loss, acc = model.evaluate_generator(test_generator, steps = 5)
    print(acc)
else:
    model = load_model('corns.h5')
    img = image.load_img(Proba, target_size=(150,150))
    tensor = image.img_to_array(img)
    tensor = np.expand_dims(tensor, axis=0)
    tensor /= 255.
    result = model.predict(tensor)
    if np.argmax(result[0]) == 0:
        print('Buckwheat')
    elif np.argmax(result[0]) == 1:
        print ('Millet')
    elif np.argmax(result[0]) == 2:
        print ('Pasta')
    else:
        print('Rice')
