from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2

# Define a simpler pure CNN model
model1 = Sequential([
    # Input Layer
    Input(shape=(3749, 2), name="inputs_cnn"),

    # First Convolutional Block
    Conv1D(filters=32, kernel_size=7, activation='relu', kernel_regularizer=l2(0.001), name="conv1d_1"),
    BatchNormalization(name="batch_norm_1"),
    MaxPooling1D(pool_size=3, name="max_pooling1d_1"),
    Dropout(0.3, name="dropout_1"),

    # Second Convolutional Block
    Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=l2(0.001), name="conv1d_2"),
    BatchNormalization(name="batch_norm_2"),
    MaxPooling1D(pool_size=3, name="max_pooling1d_2"),
    Dropout(0.3, name="dropout_2"),

    # Third Convolutional Block
    Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001), name="conv1d_3"),
    BatchNormalization(name="batch_norm_3"),
    MaxPooling1D(pool_size=2, name="max_pooling1d_3"),
    Dropout(0.4, name="dropout_3"),

    # Flatten and Fully Connected Layers
    Flatten(name="flatten"),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001), name="dense_1"),
    Dropout(0.5, name="dropout_4"),
    Dense(5, activation='softmax', name="main_output")  # Adjust the output size for the number of classes
])

# Compile the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model1.summary()
