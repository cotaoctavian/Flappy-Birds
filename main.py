from ple import PLE
from ple.games.flappybird import FlappyBird
from Network import Network
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

game_width = 288
game_height = 512
game_pipe_gap = 100


# Run Keras on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_gap_size(y_bottom, y_top):
    return y_bottom - y_top


def get_reward_relative_to_pipe(y_bird, y_bottom, y_top, delta_x, max_width):
    gap_size = get_gap_size(y_bottom, y_top)
    delta_y = np.absolute(y_bird - (y_top + gap_size / 2))
    reward_for_getting_inside_the_gap = (gap_size / 2) - delta_y

    # if delta_x > max_width:
    #     delta_x = max_width

    # reward_weight = (max_width - delta_x) / max_width

    return reward_for_getting_inside_the_gap


def get_reward(state, first_pipe_importance=0.9):
    return first_pipe_importance * get_reward_relative_to_pipe(state['player_y'],
                                                               state['next_pipe_bottom_y'],
                                                               state['next_pipe_top_y'],
                                                               state['next_pipe_dist_to_player'],
                                                               game_width) + \
           (1 - first_pipe_importance) * get_reward_relative_to_pipe(state['player_y'],
                                                                     state['next_next_pipe_bottom_y'],
                                                                     state['next_next_pipe_top_y'],
                                                                     state['next_next_pipe_dist_to_player'],
                                                                     game_width)


def q_learning(file_name=None, gamma=0.9, epsilon=0.5):
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = FlappyBird(width=game_width, height=game_height, pipe_gap=game_pipe_gap)

    p = PLE(game)
    p.init()

    last_state = None
    last_action = 0
    last_actions_q_values = [0, 0]
    states_buffer = []
    labels_buffer = []
    counter = 0

    network = Network()
    if file_name is not None:
        network.load(file_name)
    else:
        network.create_layers(activation_hidden_layers="relu",
                              activation_last_layer="linear",
                              weight_initializer="glorot_uniform",
                              bias_initializer="glorot_uniform",
                              loss_function="binary_crossentropy",
                              optimizer="Adadelta")  # creating layers

    while 1:
        if p.game_over():
            p.reset_game()
            counter = 0
            # plt.pause(0.01)
            # train network
            if states_buffer[0] is None:
                states_buffer = states_buffer[1:]
                labels_buffer = labels_buffer[1:]
            network.train(x=states_buffer, y=labels_buffer)
            network.save_file()

            states_buffer.clear()
            labels_buffer.clear()

            if epsilon > 0.1:
                epsilon = epsilon - 0.00001


        current_state = p.getGameState()
        # plt.scatter(counter, game_height - current_state['player_y'])

        actions_q_values = network.Q(current_state.values())
        actions_indexes = np.arange(len(actions_q_values))

        optimal_action_to_take = np.argmax(actions_q_values)
        random_action = np.random.choice(actions_indexes)

        if np.random.uniform() < epsilon:
            action = random_action
        else:
            action = optimal_action_to_take

        reward = get_reward(state=current_state)
        max_q = max(actions_q_values)

        label = last_actions_q_values
        label[last_action] = reward + gamma * max_q

        states_buffer += [last_state]
        labels_buffer += [label]

        p.act(None if action == 0 else 119)

        last_action = action
        last_state = current_state
        last_actions_q_values = actions_q_values

        sys.stdout.write(f"\rThe bird's' score is: {reward}.")
        sys.stdout.flush()

        counter += 1


def play(file_name, number_of_games=1):
    game = FlappyBird(width=game_width, height=game_height, pipe_gap=game_pipe_gap)

    p = PLE(game, display_screen=True, force_fps=False)
    p.init()

    network = Network()
    network.load(file_name)

    for i in range(number_of_games):
        if i > 0:
            p.reset_game()
        while not p.game_over():
            state = p.getGameState()
            actions_q_values = network.Q(state.values())
            action_taken_index = np.argmax(actions_q_values)

            p.act(None if action_taken_index == 0 else 119)


option = input('Do you want to train me or see me play? (Write "learn" or "play")\n')
while option.lower() not in ['play', 'learn']:
    option = input('Write "learn" or "play"\n')
if option.lower() == 'learn':
    file = input('Where should I get the weights from?(leave empty for new network)\n')
    if file == "":
        file = None
    q_learning(file)
else:
    file = input('Where should I get the weights from?\n')
    number_of_games_to_play = input('How many games should I play?\n')
    play(file, int(number_of_games_to_play))

#sdasda