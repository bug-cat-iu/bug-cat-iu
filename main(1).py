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
import math
import gc
seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_episodes1(opt, sce, ue,uavs, scenario):
    values = 0
    BS = scenario.Get_BaseStations()
    save_state = [tuple(BS[i].Get_Location().tolist()) for i in range(sce.nUAV_BS)]
    action_uav = torch.zeros(sce.nUAV_BS)
    next_state = [None] * sce.nUAV_BS
    new_state = [None] * sce.nUAV_BS
    reset_state = Create_state(scenario, ue, sce)  # 用于重置状态
    state = Create_state(scenario, ue, sce)# 初始化返回
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)
    step = 0
    save_result = []
    for nepisode in range(opt.nepisodes):
          # 画图时使用
        goto = False
        initial_state_reward = Get_reward(opt, sce, scenario, ue)
        reset_values = initial_state_reward
        reward = torch.zeros(sce.nUAV_BS)
        reward_mat = torch.zeros(sce.nUAV_BS)
        while True:
            gc.collect()
            epslion = epsilon_by_frame(step)
            # 要求探索率随着迭代次数增加而减小
            for i in range(sce.nUAV_BS):
                action_uav[i] = uavs[i].Select_Action(epslion)  # Select action
            for i in range(sce.nUAV_BS):
                new_state[i] = state
                uavs[i].state = uavs[i].change_state(action_uav[i], uavs[i].state)
                scenario = update_UAV(sce, uavs)
                next_state[i] = Create_state(scenario, ue, sce)
                uavs[i].observation_space = next_state[i]
                values = Get_reward(opt, sce, scenario, ue)
                state = next_state[i]
                junge, change = Junge(uavs[i].state,sce)
                if values>initial_state_reward:
                    reward[i] = 1
                    initial_state_reward = values
                elif values==initial_state_reward:
                    reward[i] = (values-reset_values)/100
                else:
                    reward[i] = 0
                    # uavs[i].state = new_state[i][0][i*3:(i+1)*3]
                    # cenario = update_UAV(sce, uavs)
                    # state = Create_state(scenario, ue, sce)
                if junge:
                    reward[i] = -1
                    for m in range(sce.nUAV_BS):  # 重置状态
                        for j in range(3):
                            uavs[m].state[j] = save_state[m][j]
                    scenario = update_UAV(sce, uavs)
                    state = Create_state(scenario, ue, sce)
                    save_result.append([re.item() for re in reward_mat])
                    reward_mat = torch.zeros(sce.nUAV_BS)
                    initial_state_reward = reset_values
                    # uavs[i].state = uavs[i].change_state(change, uavs[i].state)
                    # scenario = update_UAV(sce, uavs)
                    # state = Create_state(scenario, ue, sce)
                reward_mat[i] += reward[i]
            for i in range(sce.nUAV_BS):
                uavs[i].Save_Transition(new_state[i], action_uav[i], next_state[i],
                                        reward[i])  # Save the state transition
                uavs[i].Optimize_Model()  # Train the model
                if step % opt.nupdate == 0:  # Update the target network for a period
                    uavs[i].Target_Update()
            step +=1
            if step % 100 == 0 :
                fig, ax = plt.subplots()
                x = range(len(save_result))
                mat_reward = []
                epolis = len(save_result)
                for col in range(np.array(save_result).shape[1]):
                    mat_reward.append(np.array(save_result)[:, col])
                for i in range(sce.nUAV_BS):
                    ax.plot(x, mat_reward[i], label='uav_{}'.format(i))
                ax.set_xlabel('epolisn+'
                              '')  # 设置x轴名称 x label
                ax.set_ylabel('reward')  # 设置y轴名称 y label
                ax.set_title('Simple Plot')  # 设置图名为Simple Plot
                ax.legend()  # 自动检测要在图例中显示的元素，并且显示
                plt.savefig("reward_eplison.jpg")
                plt.show()  # 图形可视化
            # print(f"{nstep}次", len(save_result), save_result[0], save_result[-1], "mean:", np.mean(save_result), "max:",
            #       max(save_result), "min:", min(save_result), "epolisv:", epsilon_by_frame)


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

def Get_reward(opt,sce,scenario,ue):
    reward = 0.0
    ue_loalond = []
    save_state = []
    save_SIRN = []
    Noise = 10 ** ((sce.N0) / 10) * sce.BW
    action = torch.zeros(opt.nue, dtype=int)
    for i in range(opt.nue):
        ue_loalond.append(ue[i].Get_Location())  # 获取所有ue位置
    for i in range(opt.nue):
        action[i] = ue[i].Select_Action(scenario, sce)
        save_state.append(action[i])
    for i in range(opt.nue):
        save_SIRN.append(ue[i].Get_Reward(ue_loalond, action[i], scenario, opt, sce))  # 获得奖励
    for i in range(opt.nue):
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

def create_ue(opt, sce, scenario, device):
    ue = []  # 代理矢量
    for i in range(opt.nue):
        ue.append(UE(opt, sce, scenario, index=i, device=device))  # 初始化, 为每个代理创建CNET
    return ue

def update_UAV(sce,uavs):
    state = []
    for uav in uavs:
        state.append(uav.state)
    return Scenario(sce,state)

def Create_state(scenario,ue,sce):
    state = []
    BS = scenario.Get_BaseStations()
    for i in range(scenario.BS_Number()):
        for j in range(3):
            state.append(BS[i].Get_Location()[j])
    for i in range(len(ue)):
        for j in range(3):
            state.append(ue[i].Get_Location()[j])
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
    s = create_ue(opt, sce, scenario, device)  # 初始化
    run_episodes1(opt, sce, s,uavs,scenario)

def Associative_assignment(opt,sce,scenario,uavs,ue):
    global_step = 0
    nepisode = 0
    action = torch.zeros(opt.nue, dtype=int)
    reward = torch.zeros(opt.nue)
    QoS = torch.zeros(opt.nue)
    state_target = torch.ones(opt.nue)  # The QoS requirement
    while nepisode < opt.nepisodes:
        state = torch.zeros(opt.nue)  # Reset the state   
        next_state = torch.zeros(opt.nue)  # Reset the next_state
        nstep = 0
        while nstep < opt.nsteps:
            eps_threshold = opt.eps_min + opt.eps_increment * nstep * (nepisode + 1)
            if eps_threshold > opt.eps_max:
                eps_threshold = opt.eps_max  # Linear increasing epsilon
                # eps_threshold = opt.eps_min + (opt.eps_max - opt.eps_min) * np.exp(-1. * nstep * (nepisode + 1)/opt.eps_decay) 
                # Exponential decay epsilon
            for i in range(opt.nue):
                action[i] = ue[i].Select_Action(state, scenario, eps_threshold)  # Select action
            for i in range(opt.nue):
                QoS[i], reward[i] = Get_reward(opt,sce,scenario,ue)
                next_state[i] = QoS[i]
            for i in range(opt.nue):
                ue[i].Save_Transition(state, action[i], next_state, reward[i],
                                          scenario)  # Save the state transition
                ue[i].Optimize_Model()  # Train the model
                if nstep % opt.nupdate == 0:  # Update the target network for a period
                    ue[i].Target_Update()
            state = copy.deepcopy(next_state)  # State transits 
            if torch.all(state.eq(state_target)):  # If QoS is satisified, break
                break
            nstep += 1
        print('Episode Number:', nepisode, 'Training Step:', nstep)
        #   print('Final State:', state)
        nepisode += 1

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