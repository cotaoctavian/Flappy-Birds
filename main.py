from ple import PLE
from ple.games.flappybird import FlappyBird
import os
import numpy as np

game_width = 288
game_height = 512
game_pipe_gap = 100


def get_gap_size(y_bottom, y_top):
    return y_bottom - y_top


def get_reward(y_bird, y_bottom, y_top, delta_x, max_width):
    gap_size = get_gap_size(y_bottom, y_top)
    delta_y = np.absolute(y_bird - (y_top + gap_size / 2))
    reward_for_getting_inside_the_gap = 1 - (delta_y / (gap_size / 2))

    if delta_x > max_width:
        delta_x = max_width

    reward_weight = (max_width - delta_x) / max_width

    return reward_weight * reward_for_getting_inside_the_gap


def train(first_pipe_importance=0.9):
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = FlappyBird(width=game_width, height=game_height, pipe_gap=game_pipe_gap)

    p = PLE(game, fps=30, display_screen=False, force_fps=False)
    p.init()

    actions = list(map(lambda x: 1 if x is not None else 0, p.getActionSet()))
    last_state = None
    current_state = None
    action_taken = None
    reward = None

    while 1:
        if p.game_over():
            p.reset_game()
        last_state = p.getGameState()
        # Call the nn with last_state
        # action_taken = argmax(resultats from the nn)
        p.act(action_taken)
        current_state = p.getGameState()  # it will be equal with the new state instead
        reward = first_pipe_importance * get_reward(current_state['player_y'],
                                                    current_state['next_pipe_bottom_y'],
                                                    current_state['next_pipe_top_y'],
                                                    current_state['next_pipe_dist_to_player'],
                                                    game_width) + \
                 (1 - first_pipe_importance) * get_reward(current_state['player_y'],
                                                          current_state['next_next_pipe_bottom_y'],
                                                          current_state['next_next_pipe_top_y'],
                                                          current_state['next_next_pipe_dist_to_player'],
                                                          game_width)
        print(last_state, action_taken, reward, current_state)


def play(weights_file_name, number_of_games=1):
    game = FlappyBird(width=game_width, height=game_height, pipe_gap=game_pipe_gap)

    p = PLE(game, fps=30, display_screen=True, force_fps=False)
    p.init()

    # TODO: get the actor from the network

    for i in range(number_of_games):
        if i > 0:
            p.reset_game()
        while not p.game_over():
            state = p.getGameState()
            p.getActionSet()

            # TODO: get the actor decide how to act
            # p.act()


option = input('Do you want to train me or see me play? (Write "train" or "play")\n')
while option.lower() not in ['play', 'train']:
    option = input('Write "train" or "play"\n')
if option.lower() == 'train':
    train()
else:
    file_name = input('Where should I get the weights from?\n')
    number_of_games_to_play = input('How many games should I play?\n')
    play(file_name, int(number_of_games_to_play))
