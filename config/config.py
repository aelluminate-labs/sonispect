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

    # NN-specific configurations
    FEATURES_FILE_NN = "models/nn_model.pkl"
    TRAIN_SPLIT_NN = 0.8
    NN_EPOCHS = 50
    NN_BATCH_SIZE = 32
    NN_LEARNING_RATE = 0.001
    NN_HIDDEN_UNITS = [128, 64]
    NUM_CLASSES = 10