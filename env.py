import numpy as np
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 20  # grid size
ENV_H = 36  # env height
ENV_W = ENV_H  # env width
halfUnit = UNIT / 2
obsNum = 10
color = ['lightblue', 'pink', 'royalblue', 'pink', 'lightblue', 'lightblue']
class ENV(tk.Tk, object):
    def __init__(self, agentNum):
        super(ENV, self).__init__()
        self.agentNum = agentNum
        self.ENV_H = ENV_H
        self.obsNum = obsNum
        self.n_actions_ts = self.agentNum
        self.historyStep = 1
        self.n_states_ts = 2 * (2 * self.agentNum-1) + 2*(self.agentNum-1)*self.historyStep*2
        self.n_states_ca = 7  # number of detecting directions
        self.n_actions_ca = 1
        self.max_torque = np.pi / 2
        self.stepLength = UNIT*0.5
        self.stepLengthFree = UNIT*1
        self.observeRange = 4  # detection radius
        self.observeTimes = 400
        self.agent_all = [None] * self.agentNum
        self.target_all = [None] * self.agentNum
        self.obstacle_all = [None] * obsNum
        self.agentSize = 0.25*UNIT  
        self.tarSize = 0.25*UNIT  
        self.obsSize = np.ones(obsNum)
        self.agent_center = np.zeros((self.agentNum, 2))
        self.tar_center = np.zeros((self.agentNum, 2))
        self.obs_center = np.zeros((obsNum, 2))
        self.geometry('{0}x{1}'.format(ENV_H * UNIT, ENV_H * UNIT))
        self._build_env()

    def _build_env(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=(ENV_H+2) * UNIT,
                                width=(ENV_H+2) * UNIT)
        for c in range(0, ENV_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, ENV_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, ENV_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, ENV_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        self.origin = np.array([halfUnit, halfUnit])
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H - 1 + 0.01)
            self.agent_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H - 1 + 0.01)
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, self.agent_center[i, 1] + self.agentSize,
                fill='blue')
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize, self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize, self.tar_center[i, 1] + self.tarSize,
                fill='red')
        for i in range(int(obsNum/2)):  # Square obstacles
            self.obs_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H/2)
            self.obstacle_all[i] = self.canvas.create_rectangle(
                self.obs_center[i, 0] - self.obsSize[i], self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i], self.obs_center[i, 1] + self.obsSize[i],
                fill='grey')
        for i in range(int(obsNum/2), obsNum):  # Round obstacles
            self.obs_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H/2)
            self.obstacle_all[i] = self.canvas.create_oval(
                self.obs_center[i, 0] - self.obsSize[i], self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i], self.obs_center[i, 1] + self.obsSize[i],
                fill='grey')
        self.canvas.pack()

    def reset(self, agentPositionArray, tarPositionArray, obsArray, obsSize):
        self.update()
        sATAA = np.zeros((self.agentNum, 2 * (2 * self.agentNum - 1)))
        agent_coordi = np.zeros((self.agentNum, 2))
        tar_coordi = np.zeros((self.agentNum, 2))
        for i in range(self.agentNum):
            self.canvas.delete(self.agent_all[i])
            self.canvas.delete(self.target_all[i])  
        for i in range(obsNum):
            self.canvas.delete(self.obstacle_all[i])
        self.agentPositionArray = agentPositionArray
        self.tarPositionArray = tarPositionArray  
        self.obsArray = obsArray
        self.obsSize = obsSize * UNIT
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
        for i in range(int(obsNum/2)):  # Square obstacles
            self.obs_center[i] = self.origin + UNIT * self.obsArray[i]
            self.obstacle_all[i] = self.canvas.create_rectangle(
                self.obs_center[i, 0] - self.obsSize[i], self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i], self.obs_center[i, 1] + self.obsSize[i],
                fill='lightgrey', outline='lightgrey')
        for i in range(int(obsNum/2), obsNum):  # Round obstacles
            self.obs_center[i] = self.origin + UNIT * self.obsArray[i]
            self.obstacle_all[i] = self.canvas.create_oval(
                self.obs_center[i, 0] - self.obsSize[i], self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i], self.obs_center[i, 1] + self.obsSize[i],
                fill='lightgrey', outline='lightgrey')
        for i in range(self.agentNum):
            tar_coordi[i] = np.array(self.canvas.coords(self.target_all[i])[:2]) + np.array([self.tarSize, self.tarSize])
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + np.array([self.agentSize, self.agentSize])

        for i in range(self.agentNum):
            for k in range(self.agentNum):  # distances between agents and targets
                sATAA[i, 2*k: 2*(k+1)] = (tar_coordi[k] - agent_coordi[i]) / (ENV_H * UNIT)
            for j in range(self.agentNum):  # distances between agents
                if j > i:
                    sATAA[i, 2*(self.agentNum + j - 1): 2*(self.agentNum + j)] = (agent_coordi[j] - agent_coordi[i]) / (ENV_H * UNIT)
                elif j < i:
                    sATAA[i, 2*(self.agentNum + j): 2*(self.agentNum + j)+2] = - sATAA[j, 2*(self.agentNum + i - 1): 2*(self.agentNum + i)]
        return sATAA

    def detect_obstacle(self, tarAgentDirCoordi, i, otherTarCoordi):
        obstacleExist = 0
        obstacleDistance = 1
        for k in range(self.observeTimes + 1):
            observeDistance = k / self.observeTimes * self.observeRange
            observeCoordi = observeDistance * tarAgentDirCoordi
            A_coordi = self.canvas.coords(self.agent_all[i]) + UNIT * np.hstack((np.hstack(observeCoordi), np.hstack(observeCoordi)))
            for j in range(0, obsNum+1):
                if j < int(obsNum/2):  # Detecting square obstacles
                    Ob_coordi = self.canvas.coords(self.obstacle_all[j])
                    agentNonObstacle = A_coordi[2] <= Ob_coordi[0] or A_coordi[0] >= Ob_coordi[2] or A_coordi[3] <= Ob_coordi[1] or A_coordi[1] >= Ob_coordi[3]
                    if agentNonObstacle == 0:
                        obstacleExist = 1
                        obstacleDistance = observeDistance / self.observeRange
                        break
                elif j < obsNum:  # Detecting round obstacles
                    Ob_coordi = self.canvas.coords(self.obstacle_all[j])
                    agentNonObstacle = np.linalg.norm(Ob_coordi[:2]+self.obsSize[j]-A_coordi[:2]-self.agentSize) > (self.obsSize[j] + self.agentSize)
                    if agentNonObstacle == 0:
                        obstacleExist = 1
                        obstacleDistance = observeDistance / self.observeRange
                        break
                elif j == obsNum:  # Detecting targets
                    if np.linalg.norm(otherTarCoordi) != 0:
                        agentNonObstacle = np.linalg.norm(otherTarCoordi + observeCoordi)*UNIT > (self.tarSize + self.agentSize)
                        if agentNonObstacle == 0:
                            obstacleExist = 1
                            obstacleDistance = observeDistance / self.observeRange
                            break
            if agentNonObstacle == 0:
                break
        return obstacleExist, obstacleDistance

    def step_ddpg(self, action):
        base_actionA = np.array([0.0, 0.0])
        base_actionA[0] += np.sin(action) * self.stepLength
        base_actionA[1] -= np.cos(action) * self.stepLength
        return base_actionA[0], base_actionA[1]

    def step_dqn(self, action, observation, agentiDone):
        base_actionA = np.array([0.0, 0.0])
        if agentiDone != action + 1:
            base_actionA += observation[action*2: (action+1)*2]/np.linalg.norm(observation[action*2:(action+1)*2])*self.stepLengthFree
        return base_actionA[0], base_actionA[1]

    def move(self, move, agentExistObstacle_Target, otherTarCoordi, action, action_h, drawTrajectory):
        done_collision_cross = np.zeros(self.agentNum)  
        done_collision_agent = 0   
        done_collision_obs = 0
        success = 0
        arriveSame = 0
        done = np.zeros(self.agentNum)
        nextDisAA = np.zeros(int(self.agentNum*(self.agentNum-1)/2))
        sATAA = np.zeros((self.agentNum, 2*(2*self.agentNum - 1)))  
        agent_coordi = np.zeros((self.agentNum, 2))
        tar_coordi = np.zeros((self.agentNum, 2))
        agentNewPosition = np.zeros((self.agentNum, 2))
        nextAoutOb = 1  
        if drawTrajectory:
            for i in range(self.agentNum):
                self.agent_all[i] = self.canvas.create_oval(
                    self.canvas.coords(self.agent_all[i])[0], self.canvas.coords(self.agent_all[i])[1],
                    self.canvas.coords(self.agent_all[i])[2], self.canvas.coords(self.agent_all[i])[3],
                    fill=color[action[i]], outline=color[action[i]])

        for i in range(self.agentNum):
            tar_coordi[i] = np.array(self.canvas.coords(self.target_all[i])[:2]) + np.array([self.tarSize, self.tarSize])
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + np.array([self.agentSize, self.agentSize])
        sortAgent_index = np.argsort(agent_coordi[:, 0])
        disAT = np.zeros(self.agentNum)
        for i in range(self.agentNum):
            disAT[i] = np.linalg.norm(agent_coordi[i] - tar_coordi[action[i]])

        # Agents move
        for i in range(self.agentNum):
            self.canvas.move(self.agent_all[i], move[i, 0], move[i, 1])
        nextDisAT = np.zeros(self.agentNum)
        for i in range(self.agentNum):
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + np.array([self.agentSize, self.agentSize])
            nextDisAT[i] = np.linalg.norm(agent_coordi[i] - tar_coordi[action[i]])
        agentDone = [int(val) for val in np.zeros(self.agentNum)]  # agent reaches which target
        tarDone = [int(val) for val in np.zeros(self.agentNum)]    # indicates whether a target is reached
        tarChosen = [int(val) for val in np.zeros(self.agentNum)]  # indicates whether a target is selected by an agent

        reward = -1 * np.ones(self.agentNum)/ENV_H
        temp = 0
        for i in range(self.agentNum):
            for j in range(self.agentNum):
                if action[i] == j:
                    tarChosen[j] = 1
                if np.linalg.norm(agent_coordi[i] - tar_coordi[j]) < self.stepLengthFree:
                    agentDone[i] = j+1
                    tarDone[j] = 1
                    agent_coordi[i] = tar_coordi[j]
            for k in range(self.agentNum):
                if k > i:
                    nextDisAA[temp] = np.linalg.norm(agent_coordi[i] - agent_coordi [k])
                    if nextDisAA[temp] < UNIT: 
                        done_collision_cross[i], done_collision_cross[k] = 1, 1
                        if action[i] == action[k]:
                            done[i], done[k] = 1, 1
                            done_collision_agent = 1
                    temp += 1
            agentNewPosition[i] = agent_coordi[i] / UNIT - self.origin / UNIT  

        # Check collisions
        for i in range(self.agentNum):
            A_coordi = self.canvas.coords(self.agent_all[i])
            for j in range(0, obsNum+1):
                if j < int(obsNum/2):  # Square obs
                    Ob_coordi = self.canvas.coords(self.obstacle_all[j])
                    nextAoutOb = A_coordi[2] <= Ob_coordi[0] or A_coordi[0] >= Ob_coordi[2] or A_coordi[3] <= Ob_coordi[1] or A_coordi[1] >= Ob_coordi[3]
                    if nextAoutOb == 0:
                        done[i] = 1
                        break
                elif j < obsNum:  # Round obs
                    Ob_coordi = self.canvas.coords(self.obstacle_all[j])
                    nextAoutOb = np.linalg.norm(Ob_coordi[:2] + self.obsSize[j] - A_coordi[:2]- self.agentSize) > (self.obsSize[j] + self.agentSize)
                    if nextAoutOb == 0:
                        done[i] = 1
                        break
                elif j == obsNum:  # Target
                    if np.linalg.norm(otherTarCoordi[i]) != 0:  
                        nextAoutOb = np.linalg.norm(otherTarCoordi[i])*UNIT > (self.tarSize+self.agentSize)  
                        if nextAoutOb == 0:
                            done[i] = 1
                            break
            if nextAoutOb == 0:
                done_collision_obs = 1
                reward[i] -= 45/ENV_H  
                break
        for i in range(self.agentNum):
            if done_collision_agent == 1:
                break
            for j in range(self.agentNum):
                if j > i:
                    if agentDone[i] == agentDone[j] and agentDone[i] != 0:
                        if action[i] == action[j]:
                            arriveSame = 1
                            done[i], done[j] = 1, 1
                            break
        if np.sum(tarDone) == self.agentNum:  
            done = np.ones(self.agentNum)
            success = 1
            # print("***********  SUCCESS  ***********")
        conflict_num = np.zeros(self.agentNum)
        for i in range(self.agentNum):
            for k in range(self.agentNum):
                if k != i and action[i] == action[k]:
                    conflict_num[i] += 1
        for i in range(self.agentNum):
            reward[i] += -30 * conflict_num[i]/ENV_H
        if np.sum(tarChosen) == self.agentNum:
            reward += 0.8 * np.ones(self.agentNum)/ENV_H
        for i in range(self.agentNum):
            if agentExistObstacle_Target[i] and nextDisAT[i] < disAT[i]:
                reward[i] += 5*(disAT[i] - nextDisAT[i]) / UNIT / ENV_H
        # Update observation
        for i in range(self.agentNum):
            for k in range(self.agentNum):
                sATAA[i, 2 * k: 2 * (k + 1)] = (tar_coordi[k] - agent_coordi[i]) / (ENV_H * UNIT)
            temp = 0
            for j in range(self.agentNum):
                if sortAgent_index[j] != i:
                    sATAA[i, 2 * (self.agentNum + temp): 2 * (self.agentNum + temp+1)] = (agent_coordi[sortAgent_index[j]] - agent_coordi[i]) / (ENV_H * UNIT)
                    temp += 1
        return sATAA, reward, done, agentDone, done_collision_cross, done_collision_agent, done_collision_obs, success, arriveSame, agentNewPosition

    def render(self):
        self.update()
