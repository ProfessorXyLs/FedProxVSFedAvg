# date:2022/11/21 下午6:58
# author:ls
import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

example_element = next(iter(example_dataset))

preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))


def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.algorithms.build_weighted_fed_avg(

    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    # proximal_strength=0.5,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

NUM_ROUNDS = 50
logdir = "/home/ls/PycharmProjects/TFF_Image_Classification-main/FedAvg/"

summary_writer = tf.summary.create_file_writer(logdir)
state = iterative_process.initialize()

with summary_writer.as_default():
    for round_num in range(0, NUM_ROUNDS):
        result = iterative_process.next(state, federated_train_data)
        state = result.state
        metrics = result.metrics
        print('round {:2d}, metrics={}'.format(round_num, metrics))
        for name, value in metrics['client_work']['train'].items():
            tf.summary.scalar(name, value, step=round_num)


