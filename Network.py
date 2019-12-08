from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta
import tensorflow.keras.backend as K
import numpy as np


class Network:
    def __init__(self, learning_rate=0.1):
        self.model = Sequential()  # initializing the model
        self.list_file = []  # creating a part of the file name
        self.learning_rate = learning_rate

        self.created_file_name = ""
        self.created_files = False

    def create_layers(self, activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer,
                      loss_function, optimizer="", optimizer_parameters=""):

        # creating a part of the file_name
        self.list_file.extend([activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer,
                               optimizer, optimizer_parameters, loss_function])

        # second layer
        self.model.add(
            Dense(8 * 2, input_dim=8, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                  bias_initializer=bias_initializer))

        # third layer
        self.model.add(Dense(8 * 2, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

        # last layer
        self.model.add(Dense(2, activation=activation_last_layer, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

        self.model.compile(optimizer=Adadelta(lr=0.1, rho=0.95, epsilon=1e-6), loss=loss_function, metrics=None)

    def train(self, x, y):
        # convert x, y to make it work
        x = list(map(lambda d: list(d.values()), x))
        x = np.array(x)
        y = np.array(y)

        # start training
        self.model.train_on_batch(x=x, y=y)
        # K.clear_session()
        # self.model.fit(x=x, y=y, epochs=1, batch_size=1)

    def Q(self, state):
        # convert state to make it actually work
        state = list(state)
        state = np.array([np.array(state), ])

        # feed forward for Q learning to predict the actions
        # output = self.model.predict_on_batch(state)
        # output = self.model.predict(state, batch_size=1, verbose=0)
        output = K.eval(self.model(state))
        # K.clear_session()
        # return the actions
        return output[0]

    def save_file(self):
        if self.created_files is False:
            for item in self.list_file:
                if item != "":
                    self.created_file_name += str(item) + "_"
            self.created_files = True

        self.model.save(filepath=self.created_file_name + "_model.h5")
        self.model.save_weights(filepath=self.created_file_name + "_weights.h5")

    def load(self, file_name):
        self.created_file_name = file_name
        self.model = load_model(file_name + "_model.h5")
        self.model.load_weights(file_name + "_weights.h5")
