class Config:
    # Shared configurations
    DATA_DIR = "data/raw/sound_clips/"
    TRAIN_SPLIT = 0.66
    WINDOW_LENGTH = 0.020

    # KNN-specific configurations
    FEATURES_FILE_KNN = "models/knn_model.pkl"
    K_NEIGHBORS = 5

    # SVM-specific configurations
    FEATURES_FILE_SVM = "models/svm_model.pkl"
    SVM_KERNEL = "rbf"  # Options: 'linear', 'poly', 'rbf', 'sigmoid'
    SVM_C = 1.0
    SVM_GAMMA = "scale"
