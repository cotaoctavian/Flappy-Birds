import os
from ple import PLE
from ple.games.flappybird import FlappyBird

# os.putenv('SDL_VIDEODRIVER', 'fbcon')
# os.environ["SDL_VIDEODRIVER"] = "dummy"

game = FlappyBird()

p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
    if p.game_over(): #check if the game is over
        p.reset_game()
    obs = p.getGameState()
    p.act(119)
    print(obs)
    print(p.getActionSet())