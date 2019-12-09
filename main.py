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


def get_gap_size(y_bottom, y_top):
    return y_bottom - y_top


def get_reward_relative_to_pipe(y_bird, y_bottom, y_top, delta_x, max_width):
    gap_size = get_gap_size(y_bottom, y_top)
    delta_y = np.absolute(y_bird - (y_top + gap_size / 3))
    reward_for_getting_inside_the_gap = (gap_size / 3) - delta_y

    if reward_for_getting_inside_the_gap > 0:
        reward_for_getting_inside_the_gap = 5 * reward_for_getting_inside_the_gap

    if delta_x > max_width:
        delta_x = 0.9 * max_width

    reward_weight = (max_width - delta_x) / max_width

    return reward_weight * reward_for_getting_inside_the_gap


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


def q_learning(file_name=None, gamma=0.75, epsilon=0.9, buffer_size=50000, batch_size=128):
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

    network = Network()
    if file_name is not None:
        network.load(file_name)
    else:
        network.create_layers(activation_hidden_layers="relu",
                              activation_last_layer="linear",
                              weight_initializer="glorot_uniform",
                              bias_initializer="glorot_uniform",
                              loss_function="binary_crossentropy",
                              optimizer="Adadelta")

    while 1:
        if p.game_over():
            # restart the game
            p.reset_game()
            # count episodes
            episode += 1

            # update plot
            # plt.scatter(episode, last_score)
            # plt.pause(0.001)
            print(f'episode={episode}, epsilon={epsilon}')

            # adding the last entry correctly
            label = last_actions_q_values
            label[last_action] = -100
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
        reward = get_reward(state=current_state)
        max_q = max(actions_q_values)

        label = last_actions_q_values
        if current_score - last_score > 0:
            label[last_action] = current_score - last_score * 100
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

    p = PLE(game, display_screen=True, force_fps=False)
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
    q_learning(file)
else:
    file = input('Where should I get the weights from?\n')
    number_of_games_to_play = input('How many games should I play?\n')
    play(file, int(number_of_games_to_play))
