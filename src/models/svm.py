from sklearn import svm
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class SVMClassifier:
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        """
        Initialize the SVM classifier.
        :param kernel: Kernel type (e.g., 'linear', 'poly', 'rbf', 'sigmoid').
        :param C: Regularization parameter.
        :param gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        """
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, training_data: List[Tuple]):
        """
        Train the SVM model on the training dataset.
        """
        # Extract features and labels
        features, labels = self._prepare_data(training_data)
        
        # Scale the features
        features = self.scaler.fit_transform(features)
        
        # Fit the SVM model
        self.model.fit(features, labels)
        self.is_fitted = True

    def predict(self, test_data: List[Tuple]) -> List:
        """
        Predict labels for the test dataset.
        """
        if not self.is_fitted:
            raise Exception("Model must be fitted before making predictions.")
        
        # Extract features from test data
        features, _ = self._prepare_data(test_data)
        
        # Scale the features using the same scaler as training
        features = self.scaler.transform(features)
        
        # Predict using the SVM model
        return self.model.predict(features).tolist()

    @staticmethod
    def _prepare_data(data: List[Tuple]) -> Tuple[List, List]:
        """
        Prepare features and labels from the dataset.
        """
        features = [item[0] for item in data]
        labels = [item[-1] for item in data]
        return features, labels
