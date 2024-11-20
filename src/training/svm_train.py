from src.features.extractor import FeatureExtractor
from src.models.svm import SVMClassifier
from src.utils.data_loader import DataLoader
from src.utils.metrics import Metrics
from config.config import Config

def main():
    # :: Initialize feature extractor and extract features
    print("Extracting features...")
    extractor = FeatureExtractor(window_length=Config.WINDOW_LENGTH)
    extractor.process_directory(Config.DATA_DIR, Config.FEATURES_FILE_SVM, model_type="svm")
    
    # :: Load and split dataset
    print("Loading dataset...")
    training_set, test_set = DataLoader.load_dataset(Config.FEATURES_FILE_SVM, Config.TRAIN_SPLIT, model_type="svm")
    
    # :: Train SVM model and make predictions
    print("Training SVM model and making predictions...")
    classifier = SVMClassifier(kernel=Config.SVM_KERNEL, C=Config.SVM_C, gamma=Config.SVM_GAMMA)
    classifier.fit(training_set)
    predictions = classifier.predict(test_set)
    
    # :: Calculate and display accuracy
    accuracy = Metrics.accuracy(test_set, predictions)
    print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
