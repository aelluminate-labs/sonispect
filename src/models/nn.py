import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from typing import List

class NeuralNetwork:
    def __init__(self, input_dim: int, num_classes: int, hidden_units: List[int], learning_rate: float):
        self.model = Sequential()
        
        # Input layer
        self.model.add(Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in hidden_units:
            self.model.add(Dense(units, activation='relu'))
        
        # Output layer
        self.model.add(Dense(num_classes, activation='softmax'))
        
        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, X_train, y_train, epochs: int, batch_size: int, validation_data=None):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
