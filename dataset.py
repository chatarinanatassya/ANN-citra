import tensorflow as tf


class load_storage:
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_dir = 'data/data_train'
    test_dir = 'data/data_test'


    def loaddataTrainWithTensorFlow(self):
        #load data with TensorFlow
        train_data = tf.keras.preprocessing.image_dataset_from_directory(self.train_dir, image_size=(28, 28), color_mode='rgb')

        class_names = train_data.class_names
        for images, labels in train_data:
            images_1d = tf.reshape(images, (images.shape[0], -1))
            self.x_train.extend(images_1d.numpy())
            self.y_train.extend(labels.numpy())

        #convert lists to TensorFlow variables
        self.x_train = tf.convert_to_tensor(self.x_train)
        self.y_train = tf.convert_to_tensor(self.y_train)


    def loaddataTestWithTensorFlow(self):
        #load data with TensorFlow
        test_data = tf.keras.preprocessing.image_dataset_from_directory(self.test_dir, image_size=(28, 28), color_mode='rgb')

        class_names = test_data.class_names
        for images, labels in test_data:
            images_1d = tf.reshape(images, (images.shape[0], -1))
            self.x_test.extend(images_1d.numpy())
            self.y_test.extend(labels.numpy())

        #convert lists to TensorFlow variables
        self.x_test = tf.convert_to_tensor(self.x_test)
        self.y_test = tf.convert_to_tensor(self.y_test)
