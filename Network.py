from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop
import numpy as np


class Network:
    def __init__(self, learning_rate=0.1, epochs=1, mini_batch_size=128):
        self.model = Sequential()  # initializing the model
        self.batch_size = mini_batch_size
        self.epochs = epochs
        self.list_file = [self.batch_size, self.epochs]  # creating a part of the file name
        self.learning_rate = learning_rate

        self.created_files = False
        self.created_file_name = ""

    def create_layers(self, activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer):
        # creating a part of the file_name
        self.list_file.extend([activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer])

        # second layer
        self.model.add(Dense(8 * 2, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

        # third layer
        self.model.add(Dense(8 * 2, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))
        # last layer
        self.model.add(Dense(2, activation=activation_last_layer, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

    def train(self, x, y, loss_function, optimizer="", optimizer_parameters=""):
        # convert x, y to make it work
        x = list(map(lambda d: list(d.values()), x))
        x = np.array(x)
        y = np.array(y)

        # creating a part of the file_name
        if not self.created_files:
            self.list_file.extend([optimizer, optimizer_parameters, loss_function])

        # setting up the optimizer for the neuronal network
        # RMS prop
        # self.model.compile(optimizer=RMSprop(lr=0.05, rho=0.9), loss=loss_function, metrics=None)

        # SGD
        self.model.compile(optimizer=SGD(lr=0.1, momentum=0.75, nesterov=True), loss=loss_function, metrics=['acc'],
                           shuffle=True)

        # first layer + start training
        self.model.fit(x=x, y=y, epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

        self.save_file()

    def Q(self, state):
        # convert state to make it actually work
        state = list(state)
        state = np.array([np.array(state), ])

        # feed forward for Q learning to predict the actions
        output = self.model.predict(state, batch_size=None, verbose=0)

        # return the actions
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
