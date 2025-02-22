import torch 
import random 
import numpy as np
from game import SnakeGameAI, Direction,Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR= 0.001

class Agent:

    def __init__(self):
        self.n_games=0
        self.epsilon = 0 # Randomness
        self.gamma = 0.85 # Reward Discount factor 
        self.memory = deque(maxlen=MAX_MEMORY) #POPLEFT()
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model,lr=LR,gamma = self.gamma)
       

    def get_state(self,game):
        head = game.snake[0]
        # Take the 4 boxes around the head position
        point_l = Point(head.x -20,head.y)
        point_r = Point(head.x +20,head.y)
        point_u = Point(head.x,head.y -20)
        point_d = Point(head.x ,head.y+20)
        # Boolean values so that we can do AND Operation later
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger if proceeded staright
            # r d l u ->  r d l u (no change)
            (dir_r and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_d)) or 
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) 
             ,
            # Danger if proceeded right from current direction
            # r d l u -> d l u r (clockwise)
            (dir_r and game.is_collision(point_d)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) 
            ,
            # Danger if proceeded left from current direction
            # r d l u -> u r d l ( anticlockwise)
            (dir_r and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) 
            ,

            # Move direction
            dir_l,dir_d,dir_r,dir_u,

            #Food Location
            game.food.x < game.head.x , # Food is located left relative to head position
            game.food.x > game.head.x , # Food is located right relative to head position
            game.food.y < game.head.y , # Negative Y means Going Up
            game.food.y < game.head.y  # Food is located up relative to head position
        ]
        return np.array(state,dtype=int)

    def remember(self,action,state,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        # Old date gets deleted when max memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) # Returns Tuples
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)


    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # Do some random moves : Tradeoff Exploration /Exploitation
        self.epsilon = 80 - self.n_games # Explore more when inexperienced
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: #When Epsilon is a positive number
            move = random.randint(0,2)
            final_move[move] = 1
        else : 
            state0 = torch.tensor(state,dtype=torch.float)
            prediction =  self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
        plot_score = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent= Agent()
        game = SnakeGameAI()
        while True:

        #get old state ,move
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)

        # Perform new move and Get new state
            reward,done,score = game.play_step(final_move)
            state_new = agent.get_state(game)

        #Train short memory only for 1 move
            agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #Storing these data
            agent.remember(state_old,final_move,reward,state_new,done)

            if done:
                #Learn from Experiences  : Experience Replay
                game.reset()
                agent.n_games+=1
                agent.train_long_memory()

                if score > record :
                    record = score
                    agent.model.save()
                print('Game',agent.n_games,'Score',score,'Record:',record)

                # Plot
                plot_score.append(score)
                total_score+= score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_score,plot_mean_scores)    



if __name__ == '__main__': train()

