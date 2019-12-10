from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Nadam
from tensorflow.keras.layers import LeakyReLU
import tensorflow.keras.backend as K
import numpy as np
import os


class Network:
    def __init__(self, batch_size=128, gamma=0.75, epsilon=0.9, gap_division=3):
        self.model = Sequential()  # initializing the model
        self.list_file = []  # creating a part of the file name
        self.batch_size = batch_size

        self.created_file_name = "batch_size-" + str(batch_size) + "_gamma-" + str(gamma) + "_eps-" + str(epsilon) + "_gap_division-" + str(gap_division) + "_"
        self.created_files = False

    def create_layers(self, activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer,
                      loss_function, optimizer, optimizer_parameters, leaky_hidden_layers=False, leaky_last_layer=False):

        best_optimizer = None
        if optimizer.__eq__("Adadelta"):
            best_optimizer = Adadelta(optimizer_parameters['lr'], optimizer_parameters['rho'])
        elif optimizer.__eq__("SGD"):
            best_optimizer = SGD(optimizer_parameters['lr'], optimizer_parameters['momentum'], optimizer_parameters['nesterov'])
        elif optimizer.__eq__("RMSprop"):
            best_optimizer = RMSprop(optimizer_parameters['lr'], optimizer_parameters['rho'])
        elif optimizer.__eq__("Nadam"):
            best_optimizer = Nadam(optimizer_parameters['lr'], optimizer_parameters['beta_1'], optimizer_parameters['beta_2'])

        # creating a part of the file_name
        self.list_file.extend([activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer,
                               loss_function, optimizer, optimizer_parameters])

        if leaky_hidden_layers is True:
            self.model.add(Dense(8 * 2, input_dim=8, kernel_initializer=weight_initializer, bias_initializer=bias_initializer))

            self.model.add(activation_hidden_layers)

            self.model.add(Dense(16, kernel_initializer=weight_initializer, bias_initializer=bias_initializer))
            
            self.model.add(activation_hidden_layers)

        else: 
            self.model.add(Dense(8 * 2, input_dim=8, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

            self.model.add(Dense(16, activation=activation_hidden_layers, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

        if leaky_last_layer is True:
            # last layer
            self.model.add(Dense(2, kernel_initializer=weight_initializer,
                                bias_initializer=bias_initializer))

            self.model.add(activation_last_layer)
        else:
            self.model.add(Dense(2, activation=activation_last_layer, kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer))

        self.model.compile(optimizer=best_optimizer, loss=loss_function, metrics=None)

    def train(self, sample):
        # convert x, y to make it work
        x = [el[0] for el in sample]
        y = [el[1] for el in sample]

        x = list(map(lambda d: list(d.values()), x))
        x = np.array(x)
        y = np.array(y)

        self.model.train_on_batch(x=x, y=y)

        # self.save_file()

    def Q(self, state):
        # convert state to make it actually work
        state = np.array([list(state.values()), ])
        output = K.eval(self.model(state))
        return output[0]

    def save_file(self):
        if self.created_files is False:
            counter = 0
            for item in self.list_file:
                if type(item) == dict and item != "":
                    for value, key in zip(item.values(), item.keys()):
                        self.created_file_name += key + "-" + str(value) + "_"
                    self.created_file_name = self.created_file_name[:-1]
                elif type(item) != dict:
                    self.created_file_name += str(item) + "_"
                counter += 1
            self.created_files = True

        self.model.save(filepath=self.created_file_name + "_model.h5")
        self.model.save_weights(filepath=self.created_file_name + "_weights.h5")

    def load(self, file_name, rename=True):
        if rename is True:
            position, i, no_of_underscores = None, 0, 0
            for ch in file_name:
                if ch == "_":
                    no_of_underscores += 1
                if no_of_underscores == 6:
                    position = i
                    break 
                i += 1


            new_file_name = file_name[position + 1:]
            self.created_file_name += new_file_name

        self.model = load_model(file_name + "_model.h5")
        self.model.load_weights(file_name + "_weights.h5")

        if rename is True:
            os.rename(file_name + "_model.h5", self.created_file_name + "_model.h5") 
            os.rename(file_name + "_weights.h5", self.created_file_name + "_weights.h5")
