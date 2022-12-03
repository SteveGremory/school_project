import tensorflow as tf


def train_model():
    # Get the MNIST dataset
    mnist = tf.keras.datasets.mnist

    # Get the training and testing data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Get a sequential tensorflow model
    model = tf.keras.models.Sequential()
    # Flatten the input shape
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # Add the dense layer
    # Rectify leniar unit -> RELU
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    # Output layer
    # This layer makes sure that all 10 neurons add up to 1
    # Probability for each digit to be the right answer
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    # Train the model
    # EPOCHs -> How many times will the model see the same data
    model.fit(x_train, y_train, epochs=3)
    model.save("written_digits.model")
