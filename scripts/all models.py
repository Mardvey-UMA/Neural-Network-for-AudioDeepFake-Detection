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
print('Все библиотеки инициализированы')

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_dir = 'C:\\data\\train_img'
validation_dir = 'C:\\data\\val_img'
test_dir = 'C:\\data\\test_img'

train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 32, class_mode = 'categorical', target_size = (224, 224))
validation_generator = val_datagen.flow_from_directory(validation_dir, batch_size = 32, class_mode = 'categorical', target_size = (224, 224))
test_generator = test_datagen.flow_from_directory(test_dir, batch_size = 32, class_mode = 'categorical', target_size = (224, 224))

print('Создание генераторов обучения завершено')

main_path = 'C:\\Users\\NitghtWay\\PycharmProjects\\fit\\venv\\pretraindedmodels\\'
freze_path = 'C:\\Users\\NitghtWay\PycharmProjects\fit\venv\ready_models\\freez\\'
trainable_path = 'C:\\Users\\NitghtWay\PycharmProjects\\fit\\venv\\ready_models\\trainable\\'
model_names = os.listdir('C:\\Users\\NitghtWay\\PycharmProjects\\fit\\venv\\pretraindedmodels\\')
print('Модели которые будем обучать',model_names)
#mn = model_names[0]
# print(mn[:-3])
#print(os.path.join(trainable_path+mn[:-3]+'trainable.h5'))
#print(os.path.join(main_path + mn))
model_names = ['VGG19.h5']
for mn in model_names:
    lg_dr = 'C:\\Users\\NitghtWay\\PycharmProjects\\fit\\venv\\logs\\fit3\\'
    log_dir = lg_dr + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    base_model = load_model(os.path.join(main_path + mn))
    base_model.trainable = False
    model = keras.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(2, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    print('Новая модель на основе',mn[:-3],'скомпилирована')
    print('Модель имеет следующую структуру:')
    model.summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',
                                                   patience=4, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=2, verbose=1)
    print('Начало обучения модели',mn[:-3])

    history = model.fit(train_generator, batch_size=32, epochs=15, validation_data=validation_generator,
                        callbacks=[early_stopping, lr_scheduler,tensorboard_callback])
    print('Обучение закончено, запуск сохранения ')
    model.save(os.path.join(freze_path+mn[:-3]+'FREEZE.h5'))
    print('Сохранение заверщено')
    print('                      ')
    print('Получены следующие данные точности и потерь (с замороженными слоями)')
    print('Модель', mn[:-3])
    model.evaluate(test_generator)
    #text.append(model.evaluate(test_generator))
    #print(text)
    print('Разморозка слоев и повторное обучение модели',mn[:-3])
    model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    print('Новая модель на основе',mn[:-3],'скомпилирована')
    print('Модель имеет следующую структуру:')
    model.summary()
    print('Начало обучения модели', mn[:-3])
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',
    #                                                patience=4, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=2, verbose=1)
    history = model.fit(train_generator, batch_size=2, epochs=30,
                        validation_data=validation_generator, callbacks=[lr_scheduler,tensorboard_callback])
    print('Обучение закончено, запуск сохранения ')
    model.save(os.path.join(trainable_path+mn[:-3]+'trainable.h5'))
    print('Сохранение заверщено')
    print('                      ')
    print('Получены следующие данные точности и потерь (с размороженными слоями)')
    print('Модель', mn[:-3])
    model.evaluate(test_generator)
print('Обучение всех моделей завершено, не забудьте скопировать полученные метрики')
    


