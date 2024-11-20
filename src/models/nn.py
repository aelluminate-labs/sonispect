from typing import List
from keras import models, layers, optimizers, callbacks, regularizers


class NeuralNetwork:
    def __init__(self, input_dim: int, num_classes: int, hidden_units: List[int], learning_rate: float, dropout_rate: float = 0.2, l2_lamda: float = 0.1):
        self.model = models.Sequential()
        
        # :: Input layer
        self.model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in hidden_units:
            self.model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_lamda)))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        self.model.add(layers.Dense(num_classes, activation='softmax'))
        
        # Compile the model
        self.model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, X_train, y_train, epochs: int, batch_size: int, validation_data=None):
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=[early_stopping, reduce_lr])

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
