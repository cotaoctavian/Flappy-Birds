import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop
import numpy as np

# state, action, y_max
class Network:
    def __init__(self, mini_batch_size, learning_rate, epochs, file_name):
        self.model = Sequential()
        self.file_name = file_name
        self.batch_size = mini_batch_size
        self.epochs = epochs
        self.list_file = [self.batch_size, self.epochs]

    def create_layers(self, activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer):
        # second layer
        self.list_file.extend([activation_hidden_layers, activation_last_layer, weight_initializer, bias_initializer])
        self.model.add(Dense(150, activation=activation_hidden_layers, kernel_initializer=weight_initializer, bias_initializer=bias_initializer))
        self.model.add(Dense(2, activation=activation_last_layer, kernel_initializer=weight_initializer, bias_initializer=bias_initializer))

    def train(self, training_set, optimizer, optimizer_parameters, loss_function):
        # training_set = (state, label)
        self.list_file.extend([optimizer, optimizer_parameters, loss_function])
        self.model.compile(optimizer=SGD(lr=0.5, momentum=0.9, nesteros=True), loss=loss_function, metrics=['acc'], shuffle=True)
        self.model.fit((training_set[0], training_set[1]), epochs=self.epochs, batch_size=self.batch_size)
        
    def Q(self, state):
        output = self.model.predict(state, batch_size=None, verbose=0)
        return output

    def save_file(self):
        created_file_name = ""
        for item in self.list_file:
            created_file_name += str(item) + "_"
        created_file_name += ".h5"

        self.model.save(filepath="model.h5")
        self.model.save_weights(filepath=created_file_name)


if __name__ == "__main__":
    network = Network(64, 0.5, 100, "test.txt")
    network.create_layers("sigmoid", "softmax", "lecun_normal", "lecun_normal")
