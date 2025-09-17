# -*- coding:utf-8 -*-
import os

# Base directory where processed dataset will be stored
DATA_DIR = {
    'MER2023': '/mnt/merbig/dataset-process',
}

# Raw modalities (these will be filled after you run preprocessing)
PATH_TO_RAW_AUDIO = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
}
PATH_TO_RAW_FACE = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'features'),
}
PATH_TO_LABEL = {
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
}

# Tools & models
PATH_TO_PRETRAINED_MODELS = './tools'     # put pretrained models here
PATH_TO_OPENSMILE = './tools/opensmile-2.3.0'
PATH_TO_FFMPEG = 'ffmpeg'                 # system ffmpeg (installed via apt)
PATH_TO_NOISE = './tools/musan/audio-select'

# Project outputs
SAVED_ROOT = './saved'
DATA_SAVE_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')
