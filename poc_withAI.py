import os
import pygame
import argparse
import numpy as np
from poc.poc_ai import DQNAgent
from random import randint
import random
import torch.optim as optim
import torch 
import datetime
import distutils.util
DEVICE = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'

#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/100
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 250          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['train'] = False
    params["test"] = True
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params


class Game:
    """ Initialize PyGAME """
    
    def __init__(self, game_width, game_height):
        pygame.display.set_caption('SuperFishBros')
        self.game_width = game_width
        self.game_height = game_height
        self.score = 0
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("imgs/backgrounds/pirateship.jpg")
        self.running = True
        self.clock = pygame.time.Clock()
        self.player = Player(self)
        self.food = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.player)


class Player(pygame.sprite.Sprite):
     def __init__(self, game):
        super(Player, self).__init__()
        PlayerImage = pygame.image.load("imgs/players/redfish.png")
        PlayerImage = pygame.transform.scale(PlayerImage, (64, 44))
        self.game = game
        self.surf = PlayerImage.convert()
        self.surf.set_colorkey((0,0,0))
        self.rect = self.surf.get_rect()
        
     def update(self, direction):
        if direction == "up":
            self.rect.move_ip(0, -5)
        if direction == "down":
            self.rect.move_ip(0, 5)
        if direction == "left":
            self.rect.move_ip(-5, 0)
        if direction == "right":
            self.rect.move_ip(5, 0)
            # Keep player on the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self.game.game_width:
            self.rect.right = self.game.game_width
        if self.rect.top <= 0:
            self.rect.top = 0
        if self.rect.bottom >= self.game.game_height:
            self.rect.bottom = self.game.game_height        
    
class Food(pygame.sprite.Sprite):
    def __init__(self, game):
        super(Food, self).__init__()
        SalatImage = pygame.image.load("imgs/objects/collectables/meersalat.png")
        SalatImage = pygame.transform.scale(SalatImage, (40, 30))
        self.surf.set_colorkey((0,0,0))
        self.rect = self.surf.get_rect(
            center=(
                random.randint(game.game_width + 10, game.game_width + 100),
                random.randint(0, game.game_height),
            )
        )
        self.speed = random.randint(1, 5)

    # Move the sprite based on speed
    # Remove the sprite when it passes the left edge of the screen
    def update(self):
        self.rect.move_ip(-self.speed, 0)
        if self.rect.right < 0:
            self.kill()


def display_ui(game):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(game.score), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(game.bg, (10, 10))

def display(game):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game)
    for entity in game.all_sprites:
        game.gameDisplay.blit(entity.surf, entity.rect)

def update_screen():
    pygame.display.update()

def initialize_game(game, agent, batch_size):
    
    # Create a custom event for adding a new enemy
    ADDENEMY = pygame.USEREVENT + 1
    pygame.time.set_timer(ADDENEMY, 1500)

    state_init1 = agent.get_state(game.player, game.food, 3)
    action = "up"
    game.player.update(action)
    state_init2 = agent.get_state(game.player, game.food, 3)
    reward1 = agent.set_reward(game)
    agent.remember(state_init1, action, reward1, state_init2)
    agent.replay_new(agent.memory, batch_size)


def run(params):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    counter_games = 0
    while counter_games < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Initialize classes
        game = Game(400, 300)

        # Perform first move
        initialize_game(game, agent, params['batch_size'])
        if params['display']:
            display(game)
        
        steps = 0       # steps since the last positive reward
        while (game.running) and (steps < 100):
            if not params['train']:
                agent.epsilon = 0.01
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(game.player, game.food, 3)

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                final_move = random.choice(["down", "up", "left", "right"])
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(state_old.reshape((1, 6)), dtype=torch.float32).to(DEVICE)
                    prediction = agent(state_old_tensor)
                    final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]

            # perform new move and get new state
            game.player.update(final_move)
            state_new = agent.get_state(game.player, game.food, 3)

            # set reward for the new state
            reward = agent.set_reward(game)
            
            # if food is eaten, steps is set to 0
            if reward > 0:
                steps = 0
                
            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new)

            if params['display']:
                display(game)
                pygame.time.wait(params['speed'])
            steps+=1
        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])
        counter_games += 1
        # total_score += game.score
        print(f'Game {counter_games}      Score: {game.score}')
    if params['train']:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=50)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed
    if args.bayesianopt:
        bayesOpt = BayesianOptimizer(params)
        bayesOpt.optimize_RL()
    if params['train']:
        print("Training...")
        params['load_weights'] = False   # when training, the network is not pre-trained
        run(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        run(params)