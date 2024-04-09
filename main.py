# data = load_storage()
# import pandas as pd
# from keras import Sequential
import tensorflow as tf
# import ModelML
from modelml import ANNforImageClassClassification
from dataset import load_storage
# import warnings

if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    load = load_storage()
    load.loaddataTrainWithTensorFlow()
    load.loaddataTestWithTensorFlow()
    # data_xtest = load.x_test
    # print("Jumlah data uji:", len(data_xtest))
    # print("Bentuk data uji pertama:", data_xtest[0].shape)

    ML = ANNforImageClassClassification(load.x_train, load.y_train, load.x_test, load.y_test)
    print("ModelCreate")
    ML.createModel()
    print("ModelCompile")
    ML.compileModel()
    print("ModelEvaluate")
    ML.ModelEvaluate()
    print("ModelPredict")
    ML.ModelPredict(load.x_test)
    print(load.y_test)
    ML.ModelSummary()
# load_storage.loaddataTrainTF()