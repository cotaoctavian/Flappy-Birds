from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop
import numpy as np


# state, action, y_max
class Network:
    def __init__(self, learning_rate=0.1, epochs=1, mini_batch_size=128):
        self.model = Sequential()
        self.batch_size = mini_batch_size
        self.epochs = epochs
        self.list_file = [self.batch_size, self.epochs]
        self.learning_rate = learning_rate

        self.created_files = False
        self.created_file_name = ""

    def create_layers(self, activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer):
        # second layer
        self.list_file.extend([activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer])
        self.model.add(Dense(150, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))
        self.model.add(Dense(2, activation=activation_last_layer, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

    def train(self, x, y, loss_function, optimizer="", optimizer_parameters=""):
        # training_set = (state, label)
        x = list(map(lambda d: list(d.values()), x))
        x = np.array([np.array(x), ])
        y = np.array([np.array(y), ])
        if not self.created_files:
            self.list_file.extend([optimizer, optimizer_parameters, loss_function])
        self.model.compile(optimizer=SGD(lr=0.05, momentum=0.75, nesterov=True), loss=loss_function, metrics=['acc'],
                           shuffle=True)
        self.model.fit(x=x, y=y, epochs=self.epochs, batch_size=self.batch_size)

        self.save_file()

    def Q(self, state):
        state = list(state)
        state = np.array([np.array(state), ])
        output = self.model.predict(state, batch_size=None, verbose=0)
        return output[0]

    def save_file(self):
        if not self.created_files:
            for item in self.list_file:
                if item != "":
                    self.created_file_name += str(item) + "_"

        self.model.save(filepath=self.created_file_name + "model.h5")
        self.model.save_weights(filepath=self.created_file_name + "weights.h5")

        self.created_files = True

    def load(self, file_name):
        self.created_file_name = file_name
        self.created_files = True
        self.model = load_model(file_name + "_model.h5")
        self.model.load_weights(file_name + "_weights.h5")


if __name__ == "__main__":
    network = Network()
    network.create_layers("sigmoid", "softmax", "lecun_normal", "lecun_normal")
