import numpy as np
import random
from tqdm import tqdm
from scenario import Scenario
import copy, json, argparse
from dotdic import DotDic
from ue import UE
import torch
from uav import UAV
from random import seed
import matplotlib.pyplot as plt
import torch.nn.functional as F
seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_episodes1(opt, sce, agents,uavs, scenario):
    values = 0
    BS = scenario.Get_BaseStations()
    save_state = [(BS[i].Get_Location()) for i in range(sce.nUAV_BS)]
    action_uav = torch.zeros(sce.nUAV_BS)
    next_state = [None] * sce.nUAV_BS
    new_state = [None] * sce.nUAV_BS
    reward = torch.zeros(sce.nUAV_BS)
    reset_state = Create_state(scenario, agents, sce)  # 用于重置状态
    state = Create_state(scenario, agents, sce)# 初始化返回
    for nepisode in range(opt.nepisodes):
          # 画图时使用
        save_result = []
        save_result.append(Get_reward(opt, sce, scenario, agents))
        initial_state_reward = Get_reward(opt, sce, scenario, agents)
        reset_values = initial_state_reward
        for nstep in range(1,opt.nsteps+1):
            eps_threshold = opt.eps_min + opt.eps_increment * nstep * (nepisode + 1)
            if eps_threshold > opt.eps_max:
                eps_threshold = opt.eps_max  # Linear increasing epsilon
                eps_threshold = opt.eps_min + (opt.eps_max - opt.eps_min) * np.exp(-1. * nstep * (nepisode + 1)/opt.eps_decay)
                # Exponential decay epsilon
            # 要求探索率随着迭代次数增加而减小
            for i in range(sce.nUAV_BS):
                action_uav[i] = uavs[i].Select_Action(eps_threshold)  # Select action
            for i in range(sce.nUAV_BS):
                new_state[i] = state
                uavs[i].state = uavs[i].change_state(action_uav[i], uavs[i].state)
                scenario = update_UAV(sce, uavs)
                next_state[i] = Create_state(scenario, agents, sce)
                uavs[i].observation_space = next_state[i]
                values = Get_reward(opt, sce, scenario, agents)
                state = next_state[i]
                junge, change = Junge(uavs[i].state,sce)
                if values>initial_state_reward:
                    reward[i] = (values-initial_state_reward)*10
                    initial_state_reward = values
                elif values==initial_state_reward:
                    reward[i] = (values-reset_values)
                else:
                    reward[i] = (values-initial_state_reward)*10
                    uavs[i].state = new_state[i][0][i*3:(i+1)*3]
                    cenario = update_UAV(sce, uavs)
                    state = Create_state(scenario, agents, sce)
                if junge:
                    reward[i] = -100
                    uavs[i].state = uavs[i].change_state(change, uavs[i].state)
                    scenario = update_UAV(sce, uavs)
                    state = Create_state(scenario, agents, sce)
            save_result.append(values)
            for i in range(sce.nUAV_BS):
                uavs[i].Save_Transition(new_state[i], action_uav[i], next_state[i],
                                        reward[i])  # Save the state transition
                uavs[i].Optimize_Model()  # Train the model
                if nstep % opt.nupdate == 0:  # Update the target network for a period
                    uavs[i].Target_Update()
            # print(f"{nstep}次", len(save_result), save_result[0], save_result[-1], "mean:", np.mean(save_result), "max:",
            #       max(save_result), "min:", min(save_result), "epolisv:", epsilon_by_frame)
            if nstep%opt.nsteps==0:
                plt.plot(range(len(save_result)), save_result)
                plt.ylabel("values")
                plt.xlabel("epolisn")
                plt.show()
        for i in range(sce.nUAV_BS): #重置状态
            uavs[i].state = save_state[i]
        scenario = update_UAV(sce, uavs)
        state = Create_state(scenario, agents, sce)


def Junge(state,sce):
    if state[0] > sce.nUAV_BS*300:
        return True,2
    elif state[0] < -200:
        return True,3
    elif state[1] > 700:
        return True,1
    elif state[1] < -700:
        return True,0
    elif state[2] <= 0:
        return True,5
    elif state[2] > 30:
        return True,4
    return False,False

def Get_reward(opt,sce,scenario,agents):
    reward = 0.0
    agents_loalond = []
    save_state = []
    save_SIRN = []
    Noise = 10 ** ((sce.N0) / 10) * sce.BW
    action = torch.zeros(opt.nagents, dtype=int)
    for i in range(opt.nagents):
        agents_loalond.append(agents[i].Get_Location())  # 获取所有ue位置
    for i in range(opt.nagents):
        action[i] = agents[i].Select_Action(scenario, sce)
        save_state.append(action[i])
    for i in range(opt.nagents):
        save_SIRN.append(agents[i].Get_Reward(agents_loalond, action[i], scenario, opt, sce))  # 获得奖励
    for i in range(opt.nagents):
        Interference = 0.0
        new_ue = save_SIRN[i]
        for j,value in save_SIRN:
            if j == new_ue[0]:
                Interference+=value
        Interference-=new_ue[1]
        SINR = new_ue[1]/(Interference+Noise)
        if SINR < 10 ** (sce.QoS_thr / 10):SINR=0
        Rate = sce.BW * np.log2(1 + SINR)/(10**6)
        reward+=Rate
    return reward#,Counter(save_state)

def Set_uav_location(sce):
    x_bound = random.randint(0,sce.nUAV_BS*300) # 解空间范围
    y_bound = random.randint(-600,600)
    z_bound = 8
    state = np.array([x_bound,y_bound,z_bound])
    return state

def create_agents(opt, sce, scenario, device):
    agents = []  # 代理矢量
    for i in range(opt.nagents):
        agents.append(UE(opt, sce, scenario, index=i, device=device))  # 初始化, 为每个代理创建CNET
    return agents

def update_UAV(sce,uavs):
    state = []
    for uav in uavs:
        state.append(uav.state)
    return Scenario(sce,state)

def Create_state(scenario,agents,sce):
    state = []
    BS = scenario.Get_BaseStations()
    for i in range(scenario.BS_Number()):
        for j in range(3):
            state.append(BS[i].Get_Location()[j])
    for i in range(len(agents)):
        for j in range(3):
            state.append(agents[i].Get_Location()[j])
    state = np.array(state, dtype='float32')
    state = torch.from_numpy(state.reshape(1,state.shape[0]))
    return state

def creatr_UAV(sce,opt):
    uavs = []
    sce_state = []
    for i in range(sce.nUAV_BS):
        uavs.append(UAV(sce,opt,Set_uav_location(sce)))
        sce_state.append(uavs[i].state)
        uavs[i].reset_state = sce_state[i]
    return uavs,sce_state

def run_trial(opt, sce):
    uavs,sce_state = creatr_UAV(sce,opt)
    scenario = Scenario(sce, sce_state)
    s = create_agents(opt, sce, scenario, device)  # 初始化
    run_episodes1(opt, sce, s,uavs,scenario)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config_path1', type=str, help='path to existing scenarios file')
    parser.add_argument('-c2', '--config_path2', type=str, help='path to existing options file')
    parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
    args = parser.parse_args()
    sce = DotDic(json.loads(open('config_1.json', 'r').read()))
    opt = DotDic(json.loads(open('config_2.json', 'r').read()))  # 将配置文件加载为参数
    for i in range(args.ntrials):
        trial_result_path = None
        trial_opt = copy.deepcopy(opt)
        trial_sce = copy.deepcopy(sce)
        run_trial(trial_opt,trial_sce)