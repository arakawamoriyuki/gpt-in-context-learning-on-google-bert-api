import os
import sys
import traceback

import tensorflow as tf
import tensorflow_text  # noqa: F401 - load custom ops used on tf.hub
from tensorflow.keras import optimizers, callbacks
import tensorflow_hub as hub
from datasets import get_datasets


def train(model_path, tboard_path, x_train, x_test, y_train, y_test):
    class_num = y_train.shape[1]

    model = tf.keras.Sequential()
    model.add(hub.KerasLayer(
        'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
        input_shape=[],
        dtype=tf.string,
    ))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(class_num, activation='softmax'))

    model.build()

    model.summary()

    model.compile(
        optimizer=optimizers.legacy.RMSprop(
            learning_rate=0.001,
            rho=0.9,
            epsilon=None,
            decay=0.0,
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        x_train, y_train,
        epochs=1000,
        batch_size=int(len(x_train)),
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
            ),
            callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
            ),
            callbacks.TensorBoard(
                log_dir=tboard_path,
                write_graph=True,
            ),
        ],
    )

    return model


def main():
    datasets_yml_path = './input/datasets.yml'
    model_path = './models/app/1'
    tboard_path = './tboard'
    output_path = './output'

    print('Starting the training.')
    try:
        # load datasets
        x_train, x_test, y_train, y_test, labels = get_datasets(
            datasets_yml_path,
            test=True,
        )

        # training
        model = train(
            model_path,
            tboard_path,
            x_train,
            x_test,
            y_train,
            y_test,
        )

        # tensorflow v2 saved model
        tf.keras.models.save_model(model, model_path, save_format='tf')

        print('Training complete.')
    except Exception as e:
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write(f'Exception during training: {str(e)}\n{trc}')
        print(f'Exception during training: {str(e)}\n{trc}', file=sys.stderr)
        sys.exit(255)


if __name__ == '__main__':
    main()
    sys.exit(0)
