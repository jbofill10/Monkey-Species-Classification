import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd


class CustomCallBack(tfk.callbacks.Callback):

    def __init__(self, model_name):
        super().__init__()
        self.epoch_df = pd.DataFrame()
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_acc = []
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs['loss'])
        self.accuracy.append(logs['accuracy'])
        self.val_loss.append(logs['val_loss'])
        self.val_acc.append(logs['val_accuracy'])

    def on_train_end(self, logs=None):
        self.epoch_df['loss'] = self.loss
        self.epoch_df['accuracy'] = self.accuracy
        self.epoch_df['val_loss'] = self.val_loss
        self.epoch_df['val_accuracy'] = self.val_acc

        print(self.epoch_df)

        self.epoch_df.to_pickle(f'Data/pickles/{self.model_name}_epoch_df')

