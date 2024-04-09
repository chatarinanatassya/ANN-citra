import tensorflow as tf
import numpy as np

class ANNforImageClassClassification:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = []

    def createModel(self):
        #create model to train image
        input_shape = self.x_train[0].shape
        print(input_shape)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(units=120, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=2, activation='softmax') # assuming 2 classes for classification
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def compileModel(self):
        #compile model
        self.model.fit(self.x_train, self.y_train, epochs=100)

    # def ModelTest(self):
    #     self.model.evaluate(self.x_test, self.y_test, verbose=2)


    def ModelEvaluate(self):
        self.model.evaluate(self.x_test, self.y_test, verbose=2)
        # print("Model Loss : ", self.loss)
        # print("Model Accuracy : ", 1-self.loss)

    def ModelPredict(self, data):
        print("Predict")
        predictions = self.model.predict(data)
        predicted_classes = np.argmax(predictions, axis=1)
        print("Predictions : ", predicted_classes)


    def ModelSummary(self):
        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True)
        print("Model Image Saved")


    def ModelPrintParam(self):
        print(self.model.layers[1].get_weights())        

