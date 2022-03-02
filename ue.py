import numpy as np
from numpy import pi
from random import random, uniform, choice
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

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



class UE:

    def __init__(self, opt, sce, scenario, index, device):  # Initialize the agent (UE)
        self.opt = opt
        self.sce = sce
        self.id = index
        self.device = device
        self.location = self.Set_Location(scenario)

    def Set_Location(self, scenario):  # 初始化代理的位置
        _, Loc_PBS, _ = scenario.BS_Location()
        Loc_agent = np.zeros(3)
        LocM = choice(Loc_PBS)
        r = self.sce.rPBS * random()
        theta = uniform(-pi, pi)
        Loc_agent[0] = int(LocM[0] + r * np.cos(theta))
        Loc_agent[1] = int(LocM[1] + r * np.sin(theta))
        #  print(Loc_agent) 随机点的位置
        return Loc_agent

    def Get_Location(self):
        return self.location

    def Select_Action(self, scenario, sce):  # 基于网络状态选择用户的操作
        save_action = [0 for i in range(scenario.BS_Number())]
        BS = scenario.Get_BaseStations()
        for i in range(scenario.BS_Number()):  # 定义基站数量，已经忽略半径
            BS_local = BS[i].Get_Location()
            x = BS[i].Get_Location()[0] - self.location[0]
            y = BS[i].Get_Location()[1] - self.location[1]
            distance = np.sqrt((x ** 2 + y ** 2 + BS_local[2] ** 2))
            Rx_power = BS[i].Receive_Power(distance)
            if Rx_power > 0:
                save_action[i] += Rx_power
        if save_action == [0 for i in range(scenario.BS_Number())]:
            action = None
        else:
            action = save_action.index(max(save_action))
        return torch.from_numpy(np.array(action))


    def Get_Reward(self, ue_local, action, scenario,opt,sce):  # 获得行动得到的奖励
        if action == None:
            return 0
        Beam_width = math.pi/6
        Receive_Direction_gain = 2*math.pi/Beam_width-(2*math.pi-Beam_width)*0.01
        #print("action:",action,"action",action)
        dos = 1 # 自由空间的近参考距离
        rs = 1 # 波长
        ns = 1 # 路径损耗系数
        xrsoj = 1 #对数正态阴影
        oumiga = math.cos(15)
        distance = 0.0 #ue与bs距离
        Loc_diff_i = []
        saveall = 0.0
        Interference = 0.0
        BS = scenario.Get_BaseStations()
        BS_local = BS[action].Get_Location()
        x_i = BS[action].Get_Location()[0] - self.location[0]
        y_i = BS[action].Get_Location()[1] - self.location[1]
        distance_i = np.sqrt((x_i ** 2 + y_i ** 2 + BS_local[2] ** 2))
        Rx_power = BS[action].Receive_Power(distance_i) # 计算ue本身收到的功率P * 增益
        #channel_gain = 20*math.log10((4*math.pi*dos)/rs)+10*ns*math.log10(distance/dos)+xrsoj #计算信道增益
        for i in range(opt.nagents):  # Emission gain 发射增益
            x = BS[action].Get_Location()[0] - ue_local[i][0]
            y = BS[action].Get_Location()[1] - ue_local[i][1]
            distance = np.sqrt((x ** 2 + y ** 2 + BS_local[2] ** 2))
            if (x_i*x+y_i*y+BS[action].Get_Location()[2]*ue_local[2])[0]/(distance*distance_i) <= math.cos(Beam_width/2):
                Rx_power*=Receive_Direction_gain
            else:
                Rx_power*=0.01
        for j in range(scenario.BS_Number()): # Receive Direction gain 接收增益
            x_bs = BS[j].Get_Location()[0]-self.location[0]
            y_bs = BS[j].Get_Location()[1]-self.location[1]
            distance_bs = np.sqrt((x_bs ** 2 + y_bs ** 2 + BS[j].Get_Location()[2] ** 2))
            if (x_bs*x_i+y_bs*y_i+BS[j].Get_Location()[2]*self.location[2])/(distance_bs*distance) <= math.cos(Beam_width/2):
                Rx_power*=Receive_Direction_gain
            else:
                Rx_power*=0.01
        return (action,Rx_power)