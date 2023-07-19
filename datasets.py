from random import shuffle, choice

import yaml
import numpy as np
from tensorflow.keras.utils import to_categorical


def get_datasets(yml_path, test=True):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    answers = []

    with open(yml_path, 'r') as f:
        yaml_datasets = yaml.load(f, Loader=yaml.SafeLoader)
        for index, data in enumerate(yaml_datasets):
            samples = data['questions']
            answers.append(data['answer'])

            # print(f'class {index} {len(samples)} samples. ex:({samples[0]})')
            # print(f'  choice:({choice(samples)})')

            if len(samples) <= 2:
                print('Warn: サンプル数が少ない')
            if len(samples) <= 1:
                raise 'Error: サンプル数が少なくて処理不可'

            # 分類毎に1つテストデータとして利用する
            if test:
                shuffle(samples)
                x_test.append(samples.pop())
                y_test.append(index)

            for sample in samples:
                x_train.append(sample)
                y_train.append(index)

    print(f'{len(answers)} classes, {len(x_train)} samples.')

    return (
        np.array(x_train), np.array(x_test),
        to_categorical(np.array(y_train)), to_categorical(np.array(y_test)),
        np.array(answers),
    )
