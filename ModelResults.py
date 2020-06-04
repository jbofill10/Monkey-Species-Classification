import pandas as pd
import plotly.graph_objs as go


def visualize():
    epoch_df = pd.read_pickle('Data/pickles/epoch_df')

    epoch_df['epoch'] = list(map(lambda x: x+1, list(epoch_df.index)))
    fig = go.Figure(data=[
        go.Scatter(x=epoch_df['epoch'], y=epoch_df['accuracy'], line={'dash':'solid'}, name='Train'),
        go.Scatter(x=epoch_df['epoch'], y=epoch_df['val_accuracy'], line={'dash': 'solid'}, name='Validation')
    ])

    fig.update_layout(
        xaxis_title='Epochs',
        yaxis_title='Accuracy',
        title='Train Accuracy vs. Validation Accuracy - Custom Model',
        width=1200,
        height=800
    )

    fig.show()

    fig = go.Figure(data=[
        go.Scatter(x=epoch_df['epoch'], y=epoch_df['loss'], line={'dash': 'solid'}, name='Train'),
        go.Scatter(x=epoch_df['epoch'], y=epoch_df['val_loss'], line={'dash': 'solid'}, name='Validation')
    ])

    fig.update_layout(
        xaxis_title='Epochs',
        yaxis_title='Loss',
        title='Train Loss vs. Validation Loss - Custom Model',
        width=1200,
        height=800
    )

    fig.show()

    xception_epoch_df = pd.read_pickle('Data/pickles/xception_epoch_df')
    print(xception_epoch_df)
    xception_epoch_df['epoch'] = list(map(lambda x: x + 1, list(xception_epoch_df.index)))
    fig = go.Figure(data=[
        go.Scatter(x=xception_epoch_df['epoch'], y=xception_epoch_df['accuracy'], line={'dash': 'solid'}, name='Train'),
        go.Scatter(x=xception_epoch_df['epoch'], y=xception_epoch_df['val_accuracy'], line={'dash': 'solid'}, name='Validation')
    ])

    fig.update_layout(
        xaxis_title='Epochs',
        yaxis_title='Accuracy',
        title='Train Accuracy vs. Validation Accuracy - Xception',
        width=1200,
        height=800
    )

    fig.show()

    fig = go.Figure(data=[
        go.Scatter(x=xception_epoch_df['epoch'], y=xception_epoch_df['loss'], line={'dash': 'solid'}, name='Train'),
        go.Scatter(x=xception_epoch_df['epoch'], y=xception_epoch_df['val_loss'], line={'dash': 'solid'}, name='Validation')
    ])

    fig.update_layout(
        xaxis_title='Epochs',
        yaxis_title='Loss',
        title='Train Loss vs. Validation Loss - Xception',
        width=1200,
        height=800
    )

    fig.show()