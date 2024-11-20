import numpy as np
from src.utils.data_loader import DataLoader
from src.features.extractor import FeatureExtractor
from src.models.nn import NeuralNetwork
from config.config import Config

def main():
    # :: Initialize feature extractor and extract features
    print("Extracting features...")
    extractor = FeatureExtractor(window_length=Config.WINDOW_LENGTH)
    extractor.process_directory(
        directory=Config.DATA_DIR,
        output_file=Config.FEATURES_FILE_NN,
        model_type="nn"  # Specify NN model type for feature extraction
    )

    # :: Load dataset
    print("Loading dataset...")
    training_set, test_set = DataLoader.load_dataset(Config.FEATURES_FILE_NN, Config.TRAIN_SPLIT_NN, model_type="nn")
    
    # :: Prepare data
    X_train = np.array([item[0] for item in training_set]) # :: Feature vectors
    y_train = np.array([item[1] for item in training_set]) # :: Labels
    X_test = np.array([item[0] for item in test_set]) # :: Feature vectors
    y_test = np.array([item[1] for item in test_set]) # :: Labels
    
    # :: Fix labels to ensure they are within the valid range
    num_classes = Config.NUM_CLASSES
    y_train = np.clip(y_train, 0, num_classes - 1)
    y_test = np.clip(y_test, 0, num_classes - 1)

    # :: Build and train the model
    print("Training neural network...")
    input_dim = X_train.shape[1]
    nn = NeuralNetwork(input_dim, Config.NUM_CLASSES, Config.NN_HIDDEN_UNITS, Config.NN_LEARNING_RATE)
    nn.fit(X_train, y_train, epochs=Config.NN_EPOCHS, batch_size=Config.NN_BATCH_SIZE, validation_data=(X_test, y_test))

    # :: Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = nn.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()