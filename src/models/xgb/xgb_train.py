from src.helpers.extract_features import extract_features
from src.helpers.load_dataset import load_dataset
from src.helpers.normalize_features import normalize_features
from src.helpers.split_dataset import split_dataset
import xgboost as xgb
from sklearn.metrics import accuracy_score


# :: Define paths
directory = "data/raw/sound_clips/"
output_file = "models/genre_pretrained_model.bat"

# :: Extract features
instance_count = extract_features(directory, output_file, max_folders=10)

# :: Print Model Name
print("\n:: XGB Model ::")
print(f"PROCESSED: {instance_count} instances")

# :: Load dataset
features, labels = load_dataset(output_file)

# :: Normalize features
features = normalize_features(features)

# :: Split dataset
X_train, X_test, y_train, y_test = split_dataset(features, labels)

# Print the number of instances in the training and test sets
print("TRAINED: " + repr(len(X_train)) + " instances")
print("TESTED: " + repr(len(X_test)) + " instances")

# :: Train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=10,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
)
xgb_model.fit(X_train, y_train)

# :: Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# :: Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)

# :: Print the accuracy of the XGBoost model
print("ACCURACY: {:.2f}%".format(accuracy * 100))