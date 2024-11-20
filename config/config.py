class Config:
    # :: Shared configurations
    GENRE_DIR = "data/genre/"
    DATA_DIR = "data/raw/sound_clips/"
    TEST_DIR = "data/"
    TRAIN_SPLIT = 0.66
    WINDOW_LENGTH = 0.020

    # :: KNN-specific configurations
    MODEL_SAVE_PATH_KNN = "models/trained/knn_model.pkl"
    FEATURES_FILE_KNN = "models/knn_model.pkl"
    K_NEIGHBORS = 5

    # :: SVM-specific configurations
    FEATURES_FILE_SVM = "models/svm_model.pkl"
    SVM_KERNEL = "rbf"
    SVM_C = 1.0
    SVM_GAMMA = "scale"

    # :: NN-specific configurations
    MODEL_SAVE_PATH_NN = "models/trained/nn_model.h5"
    FEATURES_FILE_NN = "models/nn_model.pkl"
    TRAIN_SPLIT_NN = 0.8
    NN_EPOCHS = 50
    NN_BATCH_SIZE = 32
    NN_LEARNING_RATE = 0.001
    NN_HIDDEN_UNITS = [128, 64]
    NUM_CLASSES = 10