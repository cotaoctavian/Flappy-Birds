from ple import PLE
from ple.games.flappybird import FlappyBird
from Network import Network
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

game_width = 288
game_height = 512
game_pipe_gap = 100

# Run Keras on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def set_optimizer_parameters(optimizer):
    optimizer_parameters = dict()
    if optimizer == "Adadelta" or optimizer == "RMSprop":
        learning_rate = input("Enter learning rate (leave empty for default value (0.1)) \n")
        learning_rate = 0.1 if learning_rate == "" else float(learning_rate)

        rho = input("Enter rho value (leave empty for default value (0.95)) \n")
        rho = 0.95 if rho == "" else float(rho)

        optimizer_parameters['lr'] = learning_rate
        optimizer_parameters['rho'] = rho
    elif optimizer == "Nadam":
        learning_rate = input("Enter learning rate (leave empty for default value (0.05)) \n")
        learning_rate = 0.05 if learning_rate == "" else float(learning_rate)

        beta_1 = input("Enter beta_1 value (leave empty for default value (0.9)) \n")
        beta_1 = 0.9 if beta_1 == "" else float(beta_1)

        beta_2 = input("Enter beta_2 value (leave empty for default value (0.99)) \n")
        beta_2 = 0.99 if beta_2 == "" else float(beta_2)

        optimizer_parameters['lr'] = learning_rate
        optimizer_parameters['beta_1'] = beta_1
        optimizer_parameters['beta_2'] = beta_2
    elif optimizer == "SGD":
        learning_rate = input("Enter learning rate (leave empty for default value (0.1)) \n")
        learning_rate = 0.1 if learning_rate == "" else float(learning_rate)

        momentum = input("Enter momentum value (leave empty for default value (0.75)) \n")
        momentum = 0.75 if momentum == "" else float(momentum)

        nesterov = input("Would you like to use nesterov (leave empty for default value (True)) or (yes/no) \n")
        if nesterov == "":
            nesterov = True
        elif nesterov.lower() == "yes":
            nesterov = True
        elif nesterov.lower() == "no":
            nesterov = False

        optimizer_parameters['lr'] = learning_rate
        optimizer_parameters['momentum'] = momentum
        optimizer_parameters['nesterov'] = nesterov

    return optimizer_parameters


def get_gap_size(y_bottom, y_top):
    return y_bottom - y_top


def get_reward_relative_to_pipe(y_bird, y_bottom, y_top, delta_x, max_width, gap_division=3, reward_weight_decision=True):
    gap_size = get_gap_size(y_bottom, y_top)
    delta_y = np.absolute(y_bird - (y_top + gap_size / gap_division))
    reward_for_getting_inside_the_gap = (gap_size / gap_division) - delta_y

    if delta_x > max_width:
        delta_x = 0.9 * max_width

    reward_weight = 1
    if reward_weight_decision is True:
        reward_weight = (max_width - delta_x) / max_width

    return reward_weight * reward_for_getting_inside_the_gap


def get_reward(state, first_pipe_importance=0.9, gap_division=3, reward_weight_decision=True):
    return first_pipe_importance * get_reward_relative_to_pipe(state['player_y'],
                                                               state['next_pipe_bottom_y'],
                                                               state['next_pipe_top_y'],
                                                               state['next_pipe_dist_to_player'],
                                                               game_width,
                                                               gap_division) + \
           (1 - first_pipe_importance) * get_reward_relative_to_pipe(state['player_y'],
                                                                     state['next_next_pipe_bottom_y'],
                                                                     state['next_next_pipe_top_y'],
                                                                     state['next_next_pipe_dist_to_player'],
                                                                     game_width,
                                                                     gap_division)


def q_learning(file_name=None, plot=False, gap_division=3, gamma=0.75, epsilon=0.9, batch_size=128, reward_height_decision=True, buffer_size=50000):
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = FlappyBird(width=game_width, height=game_height, pipe_gap=game_pipe_gap)

    p = PLE(game)
    p.init()

    last_state = None
    last_action = 0
    last_actions_q_values = [0, 0]
    last_score = 0

    buffer = []
    episode = 0

    network = Network(batch_size, gamma, epsilon, gap_division)
    if file_name is not None:
        network.load(file_name)
    else:
        activation_hidden_layers = input("Enter the activation function for the hidden layers (leave empty for default activation (relu)) \n")
        activation_hidden_layers = "relu" if activation_hidden_layers == "" else activation_hidden_layers

        activation_last_layer = input("Enter the activation function for the last layer (leave empty for default activation (linear)) \n")
        activation_last_layer = "linear" if activation_last_layer == "" else activation_last_layer
        
        weight_initializer = input("Enter weight initializer (leave empty for default value (glorot_uniform)) \n")
        weight_initializer = "glorot_uniform" if weight_initializer == "" else weight_initializer

        bias_initializer = input("Enter bias initializer (leave empty for default value (glorot_uniform)) \n")
        bias_initializer = "glorot_uniform" if bias_initializer == "" else bias_initializer

        loss_func = input("Enter loss function (leave empty for default value (binary_crossentropy)) \n")
        loss_func = "binary_crossentropy" if loss_func == "" else loss_func
        
        optimizer = input("Enter the optimizer for neural network (leave empty for default value (Adadelta)) or (Adadelta/RMSprop/SGD/Nadam) \n")
        optimizer = "Adadelta" if optimizer == "" else optimizer

        optimizer_parameters = set_optimizer_parameters(optimizer)

        network.create_layers(activation_hidden_layers=activation_hidden_layers,
                              activation_last_layer=activation_last_layer,
                              weight_initializer=weight_initializer,
                              bias_initializer=bias_initializer,
                              loss_function=loss_func,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)

    while 1:
        if p.game_over():
            # restart the game
            p.reset_game()
            # count episodes
            episode += 1

            # update plot
            if plot is True:
                plt.scatter(episode, last_score)
                plt.pause(0.001)
                print(f'\n episode={episode}, score={last_score}')

            # adding the last entry correctly
            label = last_actions_q_values
            label[last_action] = -10000
            if len(buffer) < buffer_size:
                buffer += [(last_state, label)]
            else:
                buffer = buffer[1:] + [(last_state, label)]

            # reset all
            last_state = None
            last_action = 0
            last_actions_q_values = [0, 0]
            last_score = 0

        # look at the current state
        current_state = p.getGameState()
        current_score = p.score()

        # compute the actions' Q values
        actions_q_values = network.Q(current_state).tolist()

        # Compute the label for the last_state
        reward = get_reward(state=current_state, gap_division=gap_division, reward_weight_decision=reward_weight_decision)
        max_q = max(actions_q_values)

        label = last_actions_q_values
        if current_score - last_score > 0:
            label[last_action] = (current_score - last_score) * 10000
        else:
            label[last_action] = reward + gamma * max_q

        # not taking the first state into consideration
        if last_state is not None:
            # Update buffers
            if len(buffer) < buffer_size:
                buffer += [(last_state, label)]
            else:
                buffer = buffer[1:] + [(last_state, label)]

        # train
        if len(buffer) >= batch_size:
            sample = random.sample(buffer, batch_size)
            network.train(sample)

        # choose the optimal action with a chance of 1 - epsilon
        actions_indexes = np.arange(len(actions_q_values))

        optimal_action_to_take = np.argmax(actions_q_values)
        random_action = np.random.choice(actions_indexes)

        if np.random.uniform() < epsilon:
            action = random_action
        else:
            action = optimal_action_to_take

        # act accordingly
        p.act(None if action == 0 else 119)

        # update epsilon
        if epsilon > 0.1:
            epsilon = epsilon - 0.00001

        # remember everything needed from the current state
        last_action = action
        last_state = current_state
        last_actions_q_values = actions_q_values
        last_score = current_score

        # Log
        sys.stdout.write(f"\rThe bird's' score is: {reward}.")
        sys.stdout.flush()


def play(file_name, number_of_games=1):
    game = FlappyBird(width=game_width, height=game_height, pipe_gap=game_pipe_gap)

    p = PLE(game, display_screen=True, force_fps=True)
    p.init()

    network = Network()
    network.load(file_name)

    for i in range(number_of_games):
        if i > 0:
            p.reset_game()
        while not p.game_over():
            state = p.getGameState()
            actions_q_values = network.Q(state).tolist()
            action_taken_index = np.argmax(actions_q_values)

            p.act(None if action_taken_index == 0 else 119)


option = input('Do you want to train me or see me play? (Write "learn" or "play")\n')
while option.lower() not in ['play', 'learn']:
    option = input('Write "learn" or "play"\n')
if option.lower() == 'learn':
    file = input('Where should I get the weights from?(leave empty for new network)\n')
    if file == "":
        file = None

    statistics = input('Would you like to see statistics about Flappy? (yes/no)\n')
    if statistics.lower() == 'yes':
        statistics = True
    else:
        statistics = False
    
    gap_div = input('Enter gap division (leave empty for default value (3)) \n')
    if gap_div == "":
        gap_div = 3
    else:
        gap_div = int(gap_div)

    gamma = input('Enter gamma value (leave empty for default value (0.75)) \n')
    if gamma == "":
        gamma = 0.75
    else:
        gamma = float(gamma)
    
    epsilon = input('Enter epsilon value (leave empty for default value (0.9)) \n')
    if epsilon == "":
        epsilon = 0.9
    else:
        epsilon = float(epsilon)
    
    batch_size = input('Enter batch size (leave empty for default value (128)) \n')
    if batch_size == "":
        batch_size = 128
    else:
        batch_size = int(batch_size)

    reward_weight_decision = input('Would you add reward height option? (yes/no) (false by default)\n')
    if reward_weight_decision == "" or reward_weight_decision == 'no':
        reward_weight_decision = False
    elif reward_weight_decision == 'yes':
        reward_weight_decision = True

    q_learning(file, statistics, gap_div, gamma, epsilon, batch_size, reward_weight_decision)
    
else:
    file = input('Where should I get the weights from?\n')
    number_of_games_to_play = input('How many games should I play?\n')
    play(file, int(number_of_games_to_play))
