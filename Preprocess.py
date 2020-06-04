import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess():
    train_data_generator = ImageDataGenerator(
        rotation_range=30,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_data_generator.flow_from_directory(
        directory='Data/Monkeys/training/training',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    test_data_generator = ImageDataGenerator(rescale=1./255)

    test_generator = test_data_generator.flow_from_directory(
        directory='Data/Monkeys/validation/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, test_generator
