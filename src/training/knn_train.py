from src.features.extractor import FeatureExtractor
from src.models.knn import KNNClassifier
from src.utils.data_loader import DataLoader
from src.utils.metrics import Metrics
from config.config import Config

def main():
    # Initialize feature extractor and extract features
    print("Extracting features...")
    extractor = FeatureExtractor(window_length=Config.WINDOW_LENGTH)
    extractor.process_directory(Config.DATA_DIR, Config.FEATURES_FILE_KNN, model_type="knn")
    
    # Load and split dataset
    print("Loading dataset...")
    training_set, test_set = DataLoader.load_dataset(Config.FEATURES_FILE_KNN, Config.TRAIN_SPLIT, model_type="knn")
    
    # Train model and make predictions
    print("Training model and making predictions...")
    classifier = KNNClassifier(k_neighbors=Config.K_NEIGHBORS)
    classifier.fit(training_set)
    predictions = classifier.predict(test_set)
    
    # Calculate and display accuracy
    accuracy = Metrics.accuracy(test_set, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()