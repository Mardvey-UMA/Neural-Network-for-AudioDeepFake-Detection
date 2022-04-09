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

bs = 32

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_dir = 'D:\\data\\train_img'
validation_dir = 'D:\\data\\val_img'
test_dir = 'D:\\data\\test_img'

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=bs, class_mode='categorical',
                                                    target_size=(224, 224))
validation_generator = val_datagen.flow_from_directory(validation_dir, batch_size=bs, class_mode='categorical',
                                                       target_size=(224, 224))
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=bs, class_mode='categorical',
                                                  target_size=(224, 224))

base_model = tf.keras.applications.ResNet152()
base_model.trainable = False
model = keras.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(2, activation='sigmoid')
])
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
)
model.summary()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',min_delta=0.9,
                                patience=8,restore_best_weights=True)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.005,
                                patience=2,verbose=1)

history = model.fit(train_generator, batch_size=32, epochs=50, validation_data=validation_generator,
                    callbacks=[early_stopping,lr_scheduler])
model.evaluate(test_generator)
model.save('101101777.h5')


# base_model = tf.keras.applications.EfficientNetB7()
# base_model.trainable = False
# model = keras.Sequential([
#         base_model,
#         layers.Flatten(),
#         layers.Dense(2, activation='sigmoid')
# ])
# model.compile(
#         optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
#         loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#         metrics=['accuracy']
# )
# model.summary()
# history = model.fit(train_generator, batch_size=32, epochs=30, validation_data=validation_generator,
#                     callbacks=[])
# model.evaluate(test_generator)
# model.save('EfficientNetB7SGD.h5')


