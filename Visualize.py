import matplotlib.pyplot as plt
import matplotlib.style as style
import os
import pandas as pd
from PIL import Image


def visualize():
    image_paths = {}
    for dir_name, __, files in os.walk('Data/Monkeys/training/training/'):
        if dir_name == 'Data/Monkeys/training/training/': continue

        print(dir_name)

        for file_name in files:
            image_paths[dir_name[-2:]] = dir_name + '/' + file_name
            break

    labels = []
    cols = []
    with open('Data/Monkeys/monkey_labels.txt', 'r') as file:
        first = True
        for line in file.readlines():
            if len(line) == 1:
                continue
            row = line.split(",")
            row = list(map(lambda x: x.strip(), row))
            if first:
                first = False
                cols = row
                continue

            labels.append(row)

    labels_df = pd.DataFrame(labels, columns=cols)
    common_names = [i for i in labels_df['Common Name']]
    print(labels_df)

    sorted_image_paths = {k: v for k, v in sorted(image_paths.items(), key=lambda item: item[1])}
    print(sorted_image_paths)

    style.use('seaborn-poster')

    index = 1
    for key, value in sorted_image_paths.items():

        ax = plt.subplot(2, 5, index)
        ax.set_xlabel(common_names[index - 1].replace("_", " ").capitalize(), fontsize=10)
        ax.imshow(Image.open(value).resize((500,500)))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis=u'both', which=u'both', length=0)

        index += 1

    plt.tight_layout()
    plt.suptitle("Different Monkey Species", fontsize=25)
    plt.savefig('Charts/species.png')

    plt.show()


