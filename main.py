from ple import PLE
from ple.games.flappybird import FlappyBird
import os
import sys
import numpy as np
from Network import Network

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
    reward_for_getting_inside_the_gap = 1 - (delta_y / (gap_size / 2))

    if delta_x > max_width:
        delta_x = max_width

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


def q_learning(gamma=0.75, epsilon=0.9, buffer_size=500):
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = FlappyBird(width=game_width, height=game_height, pipe_gap=game_pipe_gap)

    p = PLE(game, fps=30, display_screen=False, force_fps=True)
    p.init()

    last_state = None
    last_action_taken_index = 0
    last_actions_q_values = [0.5, 0.5]
    counter = 0
    no_of_trainings = 0
    current_state = None
    action_taken = None
    reward = None
    states_buffer = []
    labels_buffer = []

    network = Network(mini_batch_size=32, epochs=3)
    network.create_layers(activation_hidden_layers="relu",
                          activation_last_layer="linear",
                          weight_initializer="glorot_uniform",
                          bias_initializer="glorot_uniform",
                          loss_function="binary_crossentropy")  # creating layers

    while 1:
        if p.game_over():
            p.reset_game()
            if len(states_buffer) > buffer_size:
                # train network
                network.train(x=states_buffer[1:], y=labels_buffer[1:])

                states_buffer.clear()
                labels_buffer.clear()

                if epsilon > 0.1:
                    epsilon = epsilon * 0.9

                no_of_trainings += 1
                counter = 0

        current_state = p.getGameState()

        actions_q_values = network.Q(current_state.values())
        action_taken_index = np.argmax(actions_q_values)

        probabilities = [(1 - epsilon), epsilon]
        actions_indexes = np.arange(len(actions_q_values))
        actions_indexes = np.delete(actions_indexes, action_taken_index)
        actions_indexes = np.append([action_taken_index], actions_indexes)

        action_taken_index = np.random.choice(actions_indexes, p=probabilities)
        action_taken = None if action_taken_index == 0 else 119

        reward = get_reward(state=current_state)
        max_q = max(actions_q_values)
        label = last_actions_q_values.copy()
        label[last_action_taken_index] = reward + gamma * max_q
        states_buffer += [last_state]
        labels_buffer += [label]

        # update
        p.act(action_taken)
        last_state = current_state.copy()
        last_action_taken_index = action_taken_index
        last_actions_q_values = actions_q_values.copy()

        counter += 1

        sys.stdout.write(f"\rGenerating training set {round((counter / buffer_size) * 100, 2)}% done. ")
        sys.stdout.flush()


def play(file_name, number_of_games=1):
    game = FlappyBird(width=game_width, height=game_height, pipe_gap=game_pipe_gap)

    p = PLE(game, fps=30, display_screen=True, force_fps=True)
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
            action_taken = None if action_taken_index == 0 else 119

            p.act(action_taken)


option = input('Do you want to train me or see me play? (Write "learn" or "play")\n')
while option.lower() not in ['play', 'learn']:
    option = input('Write "learn" or "play"\n')
if option.lower() == 'learn':
    q_learning()
else:
    file = input('Where should I get the weights from?\n')
    number_of_games_to_play = input('How many games should I play?\n')
    play(file, int(number_of_games_to_play))
