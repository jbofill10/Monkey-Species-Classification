import tensorflow as tf
import ModelInformation
from tensorflow.keras import layers, models


def run(train_gen, test_gen):
    base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

    model = models.Sequential()

    model.add(base_model)

    model.add(layers.Dense(
        units=1000,
        activation='relu'
    )
    )

    model.add(layers.Flatten())


    model.add(layers.Dense(
        units=10,
        activation='softmax'
    )
    )

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(lr=0.01),
        metrics='accuracy'
    )

    model.build(input_shape=(None, 150, 150, 3))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(lr=0.01),
        metrics=['accuracy']
    )

    callbacks = [
        ModelInformation.CustomCallBack("xception_2"),
        tf.keras.callbacks.EarlyStopping(patience=20)
    ]

    model.fit(
        x=train_gen,
        steps_per_epoch=1098 // 32,
        validation_data=test_gen,
        validation_steps=272 // 32,
        epochs=200,
        verbose=1,
        callbacks=[callbacks]
    )
