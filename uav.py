import math, random
import torch
import torch.nn as nn
from collections import namedtuple
import torch.autograd as autograd
import torch.nn.functional as F
from torch import optim
from random import sample

USE_CUDA = torch.cuda.is_available()
#将变量放到cuda上
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))  # Define a transition tuple
class ReplayMemory(object):    # Define a replay memory

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def Push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def Sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DuelingDQN(nn.Module):
    def __init__(self,observation_space,action_space):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,action_space)
        self.advantage = nn.Sequential(
            nn.Linear(action_space, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )
        self.value = nn.Sequential(
            nn.Linear(action_space, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # 定义各个层之间的关系
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        value = self.value(action)
        advantage = self.advantage(action)
        return value+advantage-advantage.mean()
        #这里不减去advantage均值的话会导致训练不稳定，因为value的作用可能有的时候被忽略掉了，有的时候又突然非常大。


    def act(self, state, epsilon):
        if random.random() <epsilon:
            with torch.no_grad():
            #如果使用的是GPU，这里需要把数据丢到GPU上
            # #.squeeze() 把数据条目中维度为1 的删除掉
                q_value = self.forward(state.type(torch.FloatTensor))
                action = q_value.max(1)[1].data[0]
            #max(1)返回每一行中最大值的那个元素，且返回其索引,max(0)是列
            #max()[1]只返回最大值的每个索引，max()[0]， 只返回最大值的每个数
                action = action.cpu().numpy()#从网络中得到的tensor形式，因为之后要输入给gym环境中，这里把它放回cpu，转为数组形式
                action = int(action)
        else:
            action = random.randrange(7)#返回指定递增基数集合中的一个随机数，基数默认值为1。
        return action


class UAV():
    def __init__(self,sce,opt,state):
        self.observation_space = []
        self.state = state
        self.action_space = 7
        self.sce = sce
        self.opt = opt
        self.reset_state = []
        self.gamma = self.opt.gamma
        self.batch_size = self.opt.batch_size
        self.epsilon = 0.9
        self.current_model = DuelingDQN((sce.nUAV_BS+sce.nPBS+sce.nFBS+opt.nagents)*3,self.action_space)
        self.target_model = DuelingDQN((sce.nUAV_BS+sce.nPBS+sce.nFBS+opt.nagents)*3,self.action_space)
        self.optimizer = optim.RMSprop(params=self.current_model.parameters(), lr=opt.learningrate,momentum=opt.momentum)
        self.memory = ReplayMemory(opt.capacity)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.target_model.eval()

    def Select_Action(self,epsilon):
        return self.current_model.act(self.observation_space,epsilon)

    def change_state(self,action,state):
        if action == 0:
            state[1] = state[1]+10
        elif action == 1:
            state[1] = state[1]-10
        elif action == 2:
            state[0] = state[0]-10
        elif action == 3:
            state[0] = state[0]+10
        elif action == 4:
            state[2] = state[2]-2
        elif action == 5:
            state[2] = state[2]+2
        elif action == 6:
            return state
        return state

    def Save_Transition(self, state, action, next_state, reward):  # +
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        # state = state.
        # next_state = next_state.unsqueeze(0)
        self.memory.Push(state, action, next_state, reward)

    def Target_Update(self):  # Update the parameters of the target network
        self.target_model.load_state_dict(self.current_model.state_dict())

    def Optimize_Model(self):
        if len(self.memory) < self.opt.batch_size:
            return
        transitions = self.memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).type(torch.int64)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.current_model(state_batch)
        state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_state_values = torch.zeros(self.opt.batch_size)
        next_action_batch = torch.unsqueeze(self.current_model(non_final_next_states).max(1)[1], 1)
        next_state_values = self.target_model(non_final_next_states).gather(1, next_action_batch)
        expected_state_action_values = (next_state_values * self.opt.gamma) + reward_batch.unsqueeze(1)
        #loss = (state_action_values - expected_state_action_values.detach()).pow(2).mean()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.current_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



