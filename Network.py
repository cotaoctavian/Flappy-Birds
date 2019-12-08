from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta
import numpy as np


class Network:
    def __init__(self, learning_rate=0.1, epochs=1, mini_batch_size=128):
        self.model = Sequential()  # initializing the model
        self.batch_size = mini_batch_size
        self.epochs = epochs
        self.list_file = [self.batch_size, self.epochs]  # creating a part of the file name
        self.learning_rate = learning_rate

        self.created_file_name = ""
        self.created_files = False

    def create_layers(self, activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer,
                      loss_function, optimizer="", optimizer_parameters=""):

        # creating a part of the file_name
        self.list_file.extend([activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer,
                               optimizer, optimizer_parameters, loss_function])

        # second layer
        self.model.add(Dense(8 * 2, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

        # third layer
        self.model.add(Dense(8 * 2, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

        # last layer
        self.model.add(Dense(2, activation=activation_last_layer, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

        # SGD
        # self.model.compile(optimizer=SGD(lr=0.1, momentum=0.75, nesterov=True), loss=loss_function, metrics=['acc'],
        #                   shuffle=True)

        # RMS prop
        # self.model.compile(optimizer=RMSprop(lr=0.05, rho=0.9), loss=loss_function, metrics=None)

        # Adadelta
        self.model.compile(optimizer=Adadelta(lr=0.1, rho=0.95, epsilon=1e-6), loss=loss_function, metrics=None)

    def train(self, x, y):
        # convert x, y to make it work
        x = list(map(lambda d: list(d.values()), x))
        x = np.array(x)
        y = np.array(y)

        # start training
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
        if self.created_files is False:
            for item in self.list_file:
                if item != "":
                    self.created_file_name += str(item) + "_"
            self.created_files = True

        self.model.save(filepath=self.created_file_name + "model.h5")
        self.model.save_weights(filepath=self.created_file_name + "weights.h5")

    def load(self, file_name):
        self.created_file_name = file_name
        self.model = load_model(file_name + "_model.h5")
        self.model.load_weights(file_name + "_weights.h5")


if __name__ == "__main__":
    network = Network()
    network.create_layers("sigmoid", "softmax", "lecun_normal", "lecun_normal")
