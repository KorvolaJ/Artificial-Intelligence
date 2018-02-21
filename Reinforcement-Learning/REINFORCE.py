# Assignment for reinforcement learning course at JKU
# objective is to make AI learn how to keep the pole balanced (200 is maximum steps)
# done by REINFORCE algorithm

import gym
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# initialize and return model
def initialize():
    model = Net()
    return model


# Reinforcement agent
class Reinforce(object):

    def __init__(self, env, alpha=0.001):
        # every agent has own model and optimizer
        self.model = initialize()
        self.env = env
        self.alpha = alpha
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        
        # place to store action probabilities
        self.log_probs = []
    
    # chooses action
    # takes state and returns index to action
    def choose_action(self, state):

        # convert to tensor
        state = torch.from_numpy(state).float().unsqueeze(0)
        # Wrap to Variable
        stateV = Variable(state)
        # get action probabilities from nn
        a_probs = self.model(stateV)
        # Using Categorical and sample() pick index to action
        # based on action probabilities
        a = Categorical(a_probs)
        action = a.sample()
        
        #save action probability
        self.log_probs.append(a.log_prob(action))
        
        return action.data[0]

        
    # this updates the parameters of network
    def update(self, states, actions, rewards, gamma=0.90):

        policy_loss = []
        rew = []   
        R = 0

        # go backwards through every step and calculate
        # reward from that point until end
        for r in rewards[::-1]:
            R = r + gamma*R
            rew.insert(0, R)
    
        rew = torch.Tensor(rew)
        # calculate REINFORCE
        rew = (rew - rew.mean()) / (rew.std() + np.finfo(np.float32).eps)
        for log_prob, re in zip(self.log_probs, rew):
            policy_loss.append(-log_prob * re)
        policy_loss = torch.cat(policy_loss).sum()
        
        # reset gradients
        self.optimizer.zero_grad()
        # compute gradients
        policy_loss.backward()
        # optimize
        self.optimizer.step()
        
        # empty log_probs
        self.log_probs = []

def generate_episode(env, agent, max_steps, render = False):

    states = []
    actions =[]
    rewards = []
    total_reward = 0.0
    # reset environment
    state = env.reset()
    for _ in range(max_steps):
        if render:
            env.render()
        # choose action and make step
        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        
        # log everything
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        # if done, break
        if done:
            break
        # switch to next state
        state = state_

    return states, actions, rewards, total_reward

def main():
    # 200 steps is maximum
    max_steps = 200
    # run for 400 episodes
    max_episodes = 400

    # select gym environment
    env = gym.make('CartPole-v0')
    # make agent
    agent = Reinforce(env, alpha = 0.0005)

    for e in range(max_episodes):
        # generate episode
        states, actions, rewards, total_reward = generate_episode(env, agent, max_steps, True)
        # update agent after episode
        agent.update(states, actions, rewards)
        print("episode: ", e ,"total reward: ", total_reward) 
        

main()