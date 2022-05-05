import numpy as np
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

ENV_H = 15  # env height
ENV_W = ENV_H  # env width
UNIT = 20  # grid size
HalfUnit = UNIT / 2
MAX_EP_STEPS = ENV_H * 2

class ENV(tk.Tk, object):
    def __init__(self, agentNum):
        super(ENV, self).__init__()
        self.ENV_H = ENV_H
        self.agentNum = agentNum
        self.n_actions = self.agentNum
        self.historyStep = 1
        self.n_features = 2 * (2 * self.agentNum - 1) + 2*(self.agentNum-1)*self.historyStep*2
        self.agent_all = [None] * self.agentNum
        self.target_all = [None] * self.agentNum
        self.agentSize = 0.25 * UNIT  
        self.tarSize = 0.5 * UNIT  
        self.agent_center = np.zeros((self.agentNum, 2))
        self.tar_center = np.zeros((self.agentNum, 2))
        self.geometry('{0}x{1}'.format(ENV_H * UNIT, ENV_H * UNIT))
        self._build_env()

    def _build_env(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=ENV_H * UNIT,
                                width=ENV_W * UNIT)
        self.origin = np.array([HalfUnit, HalfUnit])
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H - 1 + 0.01)
            self.agent_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H - 1 + 0.01)
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, self.agent_center[i, 1] + self.agentSize,
                fill='green')
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize, self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize, self.tar_center[i, 1] + self.tarSize,
                fill='orange')
        self.canvas.pack()

    def reset(self, agentPositionArray, tarPositionArray):
        self.update()
        sATAA = np.zeros((self.agentNum, 2 * (2 * self.agentNum - 1)))
        agent_coordi = np.zeros((self.agentNum, 2))
        tar_coordi = np.zeros((self.agentNum, 2))
        for i in range(self.agentNum):
            self.canvas.delete(self.agent_all[i])
            self.canvas.delete(self.target_all[i])  
        self.agentPositionArray = agentPositionArray
        self.tarPositionArray = tarPositionArray  
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + UNIT * self.tarPositionArray[i]
            self.agent_center[i] = self.origin + UNIT * self.agentPositionArray[i]
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize, self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize, self.tar_center[i, 1] + self.tarSize,
                fill='red')
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, self.agent_center[i, 1] + self.agentSize,
                fill='blue')
        for i in range(self.agentNum):
            tar_coordi[i] = np.array(self.canvas.coords(self.target_all[i])[:2]) + np.array([self.tarSize, self.tarSize])
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + np.array([self.agentSize, self.agentSize])
        for i in range(self.agentNum):
            for k in range(self.agentNum): 
                sATAA[i, 2*k: 2*(k+1)] = (tar_coordi[k] - agent_coordi[i])/(ENV_H * UNIT)
            for j in range(self.agentNum):
                if j > i:
                    sATAA[i, 2*(self.agentNum + j-1): 2*(self.agentNum + j)] = (agent_coordi[j] - agent_coordi[i])/(ENV_H * UNIT)
                elif j < i:
                    sATAA[i, 2*(self.agentNum + j): 2*(self.agentNum + j)+2] = - sATAA[j, 2*(self.agentNum + i-1): 2*(self.agentNum + i)]
        return sATAA

    def step(self, action, observation, agentiDone):
        base_actionA = np.array([0.0, 0.0])
        if agentiDone != action+1:
            base_actionA += observation[action*2:(action+1)*2] / np.linalg.norm(observation[action*2: (action+1)*2])*UNIT
        return base_actionA[0], base_actionA[1]

    def move(self, move, action):
        collision = np.zeros(self.agentNum)
        collision_true = 0
        success = 0
        conflict = 0
        nextDisAA = np.zeros(int(self.agentNum*(self.agentNum-1)/2))
        sATAA = np.zeros((self.agentNum, 2*(2*self.agentNum - 1)))  
        agent_coordi = np.zeros((self.agentNum, 2))
        tar_coordi = np.zeros((self.agentNum, 2))

        reward = -1*np.ones(self.agentNum)/ENV_H
        conflict_num = np.zeros(self.agentNum)

        for i in range(self.agentNum):
            for k in range (self.agentNum):
                if k != i and action[i] == action[k]:
                    conflict_num[i] += 1

        # Agents move
        for i in range(self.agentNum):
            self.canvas.move(self.agent_all[i], move[i, 0], move[i, 1])

        # Agents' distances from targets and other agents
        for i in range(self.agentNum):
            tar_coordi[i] = np.array(self.canvas.coords(self.target_all[i])[:2]) + np.array([self.tarSize, self.tarSize])
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + np.array([self.agentSize, self.agentSize])
        sortAgent_index = np.argsort(agent_coordi[:, 0])

        nexDisAT = np.zeros((self.agentNum, self.agentNum))
        agentDone = [int(val) for val in np.zeros(self.agentNum)]  # agent reaches which target
        tarDone = [int(val) for val in np.zeros(self.agentNum)]    # indicates whether a target is reached
        tarChosen = [int(val) for val in np.zeros(self.agentNum)]  # indicates whether a target is selected by an agent

        for i in range(self.agentNum):
            for j in range(self.agentNum):
                if action[i] == j:
                    tarChosen[j] = 1
                nexDisAT[i, j] = np.linalg.norm(agent_coordi[i] - tar_coordi[j])
                if nexDisAT[i, j] < UNIT and action[i] == j:
                    agentDone[i] = j+1
                    tarDone[j] = 1
                    agent_coordi[i] = tar_coordi[j]

        # Update observation
        for i in range(self.agentNum):
            for k in range(self.agentNum):  # distances between agents and targets
                sATAA[i, 2 * k: 2 * (k + 1)] = (tar_coordi[k] - agent_coordi[i]) / (ENV_H * UNIT)
            temp = 0
            for j in range(self.agentNum):  # distances between agents
                if sortAgent_index[j] != i:
                    sATAA[i, 2 * (self.agentNum + temp): 2 * (self.agentNum + temp+1)] = (agent_coordi[sortAgent_index[j]] - agent_coordi[i]) / (ENV_H * UNIT)
                    temp += 1

        temp = 0
        for i in range(self.agentNum):
            for k in range(self.agentNum):
                if k > i:
                    nextDisAA[temp] = np.linalg.norm(agent_coordi[i] - agent_coordi [k])
                    if nextDisAA[temp] < UNIT:
                        collision[i], collision[k] = 1, 1
                        if action[i] == action[k]:
                            collision_true = 1
                    temp += 1

        for i in range(self.agentNum):
            reward[i] += -30*conflict_num[i]/ENV_H
            for j in range(self.agentNum):
                if j > i:
                    if agentDone[i] == agentDone[j] and agentDone[i] != 0:
                        if action[i] == action[j]:
                            conflict = 1
                            reward[i] += -45/ENV_H
                            reward[j] += -45/ENV_H
                            break

        if np.sum(tarChosen) == self.agentNum:
            reward += 0.8 * np.ones(self.agentNum)/ENV_H

        if np.sum(tarDone) == self.agentNum:
            done = True
            success = 1
        elif conflict == 1 or collision_true == 1:
            done = True
        else:
            done = False

        return sATAA, reward, done, agentDone, collision, collision_true, success, conflict

    def render(self):
        self.update()
