import tensorflow as tf
import os, ModelResults, Preprocess

from Models import CNN, Xception

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    #Visualize.visualize()
    train_gen, test_gen = Preprocess.preprocess()
    #CNN.run(train_gen, test_gen)
    #Xception.run(train_gen, test_gen)
    ModelResults.visualize()

if __name__ == '__main__':
    main()
