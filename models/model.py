import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2

# Define the model architecture
def create_model():
    inputs = Input(shape=(3749, 2), name="inputs_cnn")

    # First Convolutional Block
    x = Conv1D(filters=32, kernel_size=7, activation='relu', kernel_regularizer=l2(0.001), name="conv1d_1")(inputs)
    x = BatchNormalization(name="batch_norm_1")(x)
    x = MaxPooling1D(pool_size=3, name="max_pooling1d_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)

    # Second Convolutional Block
    x = Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001), name="conv1d_2")(x)
    x = BatchNormalization(name="batch_norm_2")(x)
    x = MaxPooling1D(pool_size=3, name="max_pooling1d_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)

    # Third Convolutional Block
    x = Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), name="conv1d_3")(x)
    x = BatchNormalization(name="batch_norm_3")(x)
    x = MaxPooling1D(pool_size=2, name="max_pooling1d_3")(x)
    x = Dropout(0.4, name="dropout_3")(x)

    # Fully Connected Layers
    x = Flatten(name="flatten")(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001), name="dense_1")(x)
    x = Dropout(0.5, name="dropout_4")(x)
    outputs = Dense(5, activation='softmax', name="main_output")(x)  # Adjusted for 5-class classification

    model = Model(inputs=inputs, outputs=outputs)
    return model
