import tensorflow as tf

class Classifier(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(512, activation="relu")
        self.dense3 = tf.keras.layers.Dense(512, activation="relu")
        self.softmax = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, input_tensor, training=False):
        x = self.dense1(input_tensor)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x


