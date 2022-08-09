class Config:
    MODEL_ROOT = "/mnt/datadrive/quangdn/far/trained_models/v.0.2/all"
    LOG_ROOT = "/home/quangdn/far/face_classification/log/v.0.2"
    TRAIN_FILES = "/mnt/datadrive/quangdn/far/data/train_v0.2"
    VALID_FILES = "/mnt/datadrive/quangdn/far/data/test"

    PRETRAINED_MODEL = None

    INPUT_SIZE = (112, 112)

    RGB_MEAN = [0.5, 0.5, 0.5]
    RGB_STD = [0.5, 0.5, 0.5]
    BATCH_SIZE = 1024
    DROP_LAST = True
    LEARNING_RATE = 0.0001
    NUM_EPOCH = 200
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    NUM_EPOCH_WARM_UP = 1
    DEVICE = [0]


config = Config()
