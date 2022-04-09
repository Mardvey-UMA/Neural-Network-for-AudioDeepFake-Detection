import keras_tuner as kt
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
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
import matplotlib.pyplot as plt
import datetime

class MyTuner(kt.tuners.BayesianOptimization):
    def run_trial(self,trial,*args,**kwargs):
        kwargs['batch_size'] = trial.hp.Int('batch_size', 32, 256, step=32)
        kwargs['epochs'] = trial.hp.Int('epochs', 10, 30)


def build_model(hp):
    def optimizer_choise(opt):
        learning_rate_int = hp.Float('learning_rate', min_value=0.00001, max_value=0.01, step=0.001)
        sgd = tf.keras.optimizers.SGD(learning_rate=learning_rate_int, momentum=0.9)
        Adam = tf.keras.optimizers.Adam(learning_rate=learning_rate_int)
        rmsprop = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_int)
        if opt == 'sgd':
            return sgd
        if opt == 'Adam':
            return Adam
        if opt == 'rmsprop':
            return rmsprop

    base_model = tf.keras.applications.ResNet50()
    base_model.trainable = False
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh'])
    optimizator_choise = hp.Choice('opt', values=['sgd', 'Adam', 'rmsprop'])

    model = keras.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(2, activation=activation_choice)
    ])
    model.compile(
        optimizer=optimizer_choise(optimizator_choise),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

tuner = MyTuner(
    build_model,
    objective= 'val_accuracy',
    max_trials=100,
    directory= 'C:\\Users\\NitghtWay\\PycharmProjects\\fit\\venv\\test_models'
)
print(tuner.search_space_summary())

