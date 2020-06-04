import tensorflow as tf

from tensorflow.keras import layers, models

import ModelInformation


def run(train_gen, test_gen):

    model = models.Sequential()
    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            activation='relu'
        )
    )

    model.add(
        layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            activation='relu'
        )
    )

    model.add(
        layers.MaxPool2D(
            pool_size=(2, 2),
        )
    )

    model.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            activation='relu'
        )
    )

    model.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            activation='relu'
        )
    )

    model.add(
        layers.MaxPool2D(
            pool_size=(2, 2),
        )
    )

    model.add(layers.Flatten())

    model.add(layers.Dense(
        units=1000,
        activation='relu'
        )
    )

    model.add(layers.Dense(
        units=10,
        activation='softmax'
        )
    )

    callbacks = [
        ModelInformation.CustomCallBack("custom")
    ]

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(lr=0.01),
        metrics='accuracy'
    )

    model.build(input_shape=(None, 150, 150, 3))

    print(model.summary())

    model.fit(
        x=train_gen,
        steps_per_epoch=1098//32,
        validation_data=test_gen,
        validation_steps=272//32,
        epochs=2,
        verbose=1,
        callbacks=[callbacks]
    )

    y_pred = model.predict(test_gen)
    print(y_pred)