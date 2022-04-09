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

print('-' * 50)
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
use_early_stopping, use_tensorboard = True, True
project_path = 'D:\\доделать\\fit\\venv\\'
dataset_path = 'D:\\data\\'


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



def build_model(hp):
    def optimizer_choise(opt):
        learning_rate_int = hp.Float('learning_rate', min_value=0.00001, max_value=0.0001, step=0.001)
        sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate_int, momentum=0.9)
        Adam = tf.keras.optimizers.Adam(learning_rate=learning_rate_int)
        rmsprop = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_int)
        if opt == 'sgd':
            return sgd
        if opt == 'Adam':
            return Adam
        if opt == 'rmsprop':
            return rmsprop

    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet',
                                                )
    base_model.trainable = False
    activation_choice = hp.Choice('activation', values=['sigmoid'])
    optimizator_choise = hp.Choice('opt', values=['sgd', 'Adam', 'rmsprop'])

    model = keras.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(2, activation=activation_choice)
    ])
    model.compile(
        optimizer=optimizer_choise(optimizator_choise),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    directory=project_path + 'models\\test_models'
)

print(tuner.search_space_summary())
bs = 32
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_dir = dataset_path + 'train_img'
validation_dir = dataset_path + 'val_img'
test_dir = dataset_path + 'test_img'

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=bs, class_mode='categorical',
                                                    target_size=(224, 224))
validation_generator = val_datagen.flow_from_directory(validation_dir, batch_size=bs, class_mode='categorical',
                                                       target_size=(224, 224))
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=bs, class_mode='categorical',
                                                  target_size=(224, 224))

tuner.search(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[]
)

tuner.results_summary()

models = tuner.get_best_models(5)

n = 1
for model in models:
    model.summary()
    metrics = model.evaluate(test_generator, return_dict=True)
    acc = str(metrics["accuracy"])
    loss = str(metrics["loss"])
    model.save(str(n) +
               'acc' + acc[:4] +
               'loss' + loss[:4]+'m'+ '.h5')
    n += 1
    print()
# for model in models:
#     base_model.trainable = True
#     early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(
#         monitor="val_loss",
#         min_delta=0.1,
#         patience=10,
#         verbose=1,
#         mode="min",
#         baseline=None,
#         restore_best_weights=True,
#     )
#     print("Number of layers in the base model: ", len(base_model.layers))
#     fine_tune_at = 100
#     for layer in base_model.layers[:fine_tune_at]:
#         layer.trainable = False
#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#                   optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate / 10),
#                   metrics=['accuracy'])
#     model.summary()
#     len(model.trainable_variables)
#     fine_tune_epochs = 10
#     total_epochs = initial_epochs + fine_tune_epochs
#     history_fine = model.fit(train_generator, batch_size=32, epochs=total_epochs, validation_data=validation_generator,
#                     callbacks=[early_stopping_callbacks])
#     loss, accuracy = model.evaluate(test_generator)
#     print('Test accuracy :', accuracy)
#     model.save('nn6n23ADAM.h5')
print('Обучение завершено')
