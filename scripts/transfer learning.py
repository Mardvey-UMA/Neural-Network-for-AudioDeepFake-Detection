import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import datasets, layers, models, losses, Model
from tensorflow.keras.models import load_model
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
import matplotlib.pyplot as plt
import datetime

use_early_stopping, use_tensorboard = True, True
project_path = 'D:\\Проект на быллы ЕГЭ\\neural_network\\'
dataset_path = project_path + 'dataset\\'

def create_directory(absolute_path):
    if os.path.exists(absolute_path + 'models\\'):
        if not (os.path.exists(absolute_path + 'models\\test_models')):
            os.makedirs(absolute_path + 'models\\test_models')
        if not (os.path.exists(absolute_path + 'models\\ready_models')):
            os.makedirs(absolute_path + 'models\\ready_models')
    else:
        os.makedirs(absolute_path + 'models\\test_models')
        os.makedirs(absolute_path + 'models\\ready_models')

    if os.path.exists(absolute_path + 'log\\'):
        if not (os.path.exists(absolute_path + 'log\\fit')):
            os.makedirs(absolute_path + 'log\\fit')
    else:
        os.makedirs(absolute_path + 'log\\fit')
    if not (os.path.exists(absolute_path + 'dataset\\')):
        print('Пожалуйста создайте директорию для датасета')
    print('Все необходимые директории созданы')

def create_callbacks(es=True, tb=True):
    early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.1,
        patience=7,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=False,
    )
    lg_dr = project_path + 'log\\fit\\'
    log_dir = lg_dr + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
    if es * tb:
        return early_stopping_callbacks, tensorboard_callback
    elif es:
        return early_stopping_callbacks
    elif tb:
        return tensorboard_callback
bs = 32
base_learning_rate = 0.00001
initial_epochs = 20
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_dir = 'D:\\data\\train_img'
validation_dir = 'D:\\data\\val_img'
test_dir = 'D:\\data\\test_img'
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=bs, class_mode='categorical',
                                                    target_size=(224, 224))
validation_generator = val_datagen.flow_from_directory(validation_dir, batch_size=bs, class_mode='categorical',
                                                       target_size=(224, 224))
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=bs, class_mode='categorical',
                                                  target_size=(224, 224))
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet',
                                            )
base_model.trainable = False
model = keras.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(2, activation='sigmoid')
])
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='max',factor=0.2,
                                patience=5,verbose=1)
history = model.fit(train_generator, batch_size=32, epochs=initial_epochs, validation_data=validation_generator,
                    callbacks=[lr_scheduler])
#model.save('RNADAM.h5')

model.evaluate(test_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

n = 'y'
if n == 'y':
    base_model.trainable = True
    early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.1,
        patience=10,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )
    print("Number of layers in the base model: ", len(base_model.layers))
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
                  metrics=['accuracy'])
    model.summary()
    len(model.trainable_variables)
    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs
    history_fine = model.fit(train_generator, batch_size=32, epochs=total_epochs, validation_data=validation_generator,
                    callbacks=[early_stopping_callbacks])
    loss, accuracy = model.evaluate(test_generator)
    print('Test accuracy :', accuracy)
    model.save('nn6n23ADAM.h5')
    # acc += history_fine.history['accuracy']
    # val_acc += history_fine.history['val_accuracy']
    #
    # loss += history_fine.history['loss']
    # val_loss += history_fine.history['val_loss']
    # plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.ylim([0.8, 1])
    # plt.plot([initial_epochs - 1, initial_epochs - 1],
    #          plt.ylim(), label='Start Fine Tuning')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.ylim([0, 1.0])
    # plt.plot([initial_epochs - 1, initial_epochs - 1],
    #          plt.ylim(), label='Start Fine Tuning')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.show()

else:
    print('Обучение завершено')