# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:51:49 2020

@author: liangyu

Create the network simulation scenario
"""
import numpy as np
from numpy import pi
from random import random, uniform, choice
import math

class BS:  # 定义基站

    def __init__(self, sce, BS_index, BS_type, BS_Loc, BS_Radius):
        self.sce = sce
        self.id = BS_index
        self.BStype = BS_type
        self.BS_Loc = BS_Loc
        self.BS_Radius = BS_Radius

    def reset(self):  # 重置通道状态
        self.Ch_State = np.zeros(self.sce.nChannel)

    def Get_Location(self):
        return self.BS_Loc

    def Transmit_Power_dBm(self):  # 计算BS的发射功率
        if self.BStype == "UAV_BS":
            Tx_Power_dBm = 50
        elif self.BStype == "PBS":
            Tx_Power_dBm = 40
        elif self.BStype == "FBS":
            Tx_Power_dBm = 30
        return Tx_Power_dBm  # 在DBM中传输功率，现在没有考虑功率分配

    def Receive_Power(self, d):  # 通过传输功率和某个BS的路径丢失来计算接收的功率
        Tx_Power_dBm = self.Transmit_Power_dBm()
        if self.BStype == "UAV_BS" or self.BStype == "PBS":
            loss = 34 + 40 * np.log10(d) # 毫米波路损
        elif self.BStype == "FBS":
            loss = 37 + 30 * np.log10(d)
        if d <= self.BS_Radius:
            Rx_power_dBm = Tx_Power_dBm - loss  # 在DBM收到电力
            Rx_power = 10 ** (Rx_power_dBm / 10)  # 收到MW的电力
        else:
            Rx_power = 0.0
        return Rx_power


class Scenario():  #定义网络方案

    def __init__(self, sce,x):  # 初始化我们模拟的场景
        self.x = x
        self.sce = sce
        self.BaseStations = self.BS_Init()
        self.loc = []

    def reset(self):  # 重置我们模拟的场景
        for i in range(len(self.BaseStations)):
            self.BaseStations[i].reset()

    def BS_Number(self):
        nBS = self.sce.nUAV_BS + self.sce.nPBS + self.sce.nFBS  # 基站的数量
        return nBS

    def BS_Location(self):
        Loc_UAV_BS = np.zeros((self.sce.nUAV_BS, 3))  # 初始化BSS的位置
        Loc_PBS = np.zeros((self.sce.nPBS, 3))
        Loc_FBS = np.zeros((self.sce.nFBS, 3))

        #对基站撒点

        for i in range(self.sce.nPBS//2):
            Loc_PBS[i, 0] = 300 + 500 * i  # x-coordinate
            Loc_PBS[i, 1] = 300  # y-coordinate

        for i in range(self.sce.nPBS//2):
            Loc_PBS[i+2, 0] = 300 + 500 * i
            Loc_PBS[i+2, 1] = -300

        for i in range(self.sce.nFBS):
            Loc_FBS[i, 0] = Loc_PBS[int(i / 4), 0] + 240 * np.cos(pi / 2 * (i % 4))
            Loc_FBS[i, 1] = Loc_PBS[int(i / 4), 1] + 240 * np.sin(pi / 2 * (i % 4))

        #  对其无人机撒点，现在给了一个固定点
        for i in range(len(self.x)):
            Loc_UAV_BS[i] = self.x[i]
        #print(Loc_UAV_BS)  #查看自定义的位置
        return Loc_UAV_BS, Loc_PBS, Loc_FBS

    def BS_Init(self):  # 初始化所有基站
        BaseStations = []  # 基站矢量
        Loc_UAV_BS, Loc_PBS, Loc_FBS = self.BS_Location()

        for i in range(self.sce.nUAV_BS):  # 初始化MBSS.
            BS_index = i
            BS_type = "UAV_BS"
            BS_Loc = Loc_UAV_BS[i]
            BS_Radius = math.sqrt(abs((self.sce.rUAV_BS/2)**2-BS_Loc[2]**2))*2  # 覆盖半径随高度变化而变化
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius))

        for i in range(self.sce.nPBS):
            BS_index = self.sce.nUAV_BS + i
            BS_type = "PBS"
            BS_Loc = Loc_PBS[i]
            BS_Radius = self.sce.rPBS
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius))

        for i in range(self.sce.nFBS):
            BS_index = self.sce.nUAV_BS + self.sce.nPBS + i
            BS_type = "FBS"
            BS_Loc = Loc_FBS[i]
            BS_Radius = self.sce.rFBS
            BaseStations.append(BS(self.sce, BS_index, BS_type, BS_Loc, BS_Radius))
        return BaseStations

    def Get_BaseStations(self):
        return self.BaseStations


