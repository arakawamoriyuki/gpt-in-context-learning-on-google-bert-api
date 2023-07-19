import os
import sys
import traceback

import tensorflow as tf
import tensorflow_text  # noqa: F401 - load custom ops used on tf.hub
from tensorflow.keras import Model, layers, callbacks
import tensorflow_hub as hub
from datasets import get_datasets


class BertLayer(layers.Layer):
    def __init__(self, to_sequence=True):
        super(BertLayer, self).__init__()
        self.to_sequence = to_sequence
        self.preprocessor = hub.KerasLayer(
            'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
        )
        self.encoder = hub.KerasLayer(
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4',
            input_shape=[],
            dtype=tf.string,

            # trainable=True,
            trainable=False,
        )

    def call(self, texts):
        preprocessed_texts = self.preprocessor(texts)
        encoded_output = self.encoder(preprocessed_texts)

        if self.to_sequence:
            return encoded_output['sequence_output']
        else:
            return encoded_output['pooled_output']


def train(model_path, tboard_path, x_train, x_test, y_train, y_test):
    class_num = y_train.shape[1]

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    outputs = BertLayer()(text_input)  # outputs shape (None, 128, 768)
    node_count = outputs.shape[1] * outputs.shape[2]  # 128 * 768
    dense_output_count = outputs.shape[2] * 2  # 768 * 2
    outputs = layers.Reshape((node_count,), input_shape=outputs.shape)(outputs)
    outputs = layers.Dense(dense_output_count, activation='relu')(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dropout(0.5)(outputs)
    outputs = layers.Dense(dense_output_count, activation='relu')(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dropout(0.5)(outputs)
    outputs = layers.Dense(class_num, activation='softmax')(outputs)
    model = Model(inputs=[text_input], outputs=outputs)

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.legacy.RMSprop(
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
                patience=10,
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
