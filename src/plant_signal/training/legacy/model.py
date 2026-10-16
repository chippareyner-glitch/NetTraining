"""Small 1D CNN model for MCU-oriented time-series classification."""

from __future__ import annotations

import tensorflow as tf

from plant_signal.config import NUM_CLASSES, WINDOW_SIZE


def build_model(
    input_shape: tuple[int, int] = (WINDOW_SIZE, 1),
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    """Build the requested compact Conv1D classifier."""
    inputs = tf.keras.Input(shape=input_shape, name="signal")
    x = tf.keras.layers.Conv1D(
        filters=8,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1",
    )(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv2",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_average")(x)
    outputs = tf.keras.layers.Dense(units=num_classes, activation="softmax", name="class_probs")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="plant_signal_1d_cnn")


def main() -> None:
    model = build_model()
    model.summary()
    print(f"Total parameters: {model.count_params()}")


if __name__ == "__main__":
    main()
