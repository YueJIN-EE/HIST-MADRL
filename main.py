import numpy as np
import math
import time
from env import ENV
from algo import HIST_Alg
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--agent_num", type=int, default=0)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--pretrain_path", type=str, default='./pretrain/result')
parser.add_argument("--save_path", type=str, default='./result')
parser.add_argument("--load_path", type=str, default='./result')
args = parser.parse_args()
mode = args.mode
if mode == 'train':
    ep_num = 10000
    np.random.seed(1)
else:
    ep_num = 1000
    np.random.seed(2)
agentNum = args.agent_num
env = ENV(agentNum)
agentNum = env.agentNum
envSize = env.ENV_H
obsNum = env.obsNum
detecDirNum = env.n_states_ca
MAX_EP_STEPS = envSize * 3
envSize_ = envSize - 0.99
historyStep = env.historyStep
s_dim_dqn = env.n_states_ts
s_dim_ddpg = env.n_states_ca + 2   
s_dim = s_dim_dqn + env.n_states_ca  
n_actions = env.n_actions_ts
a_dim = env.n_actions_ca
a_bound = env.max_torque
model_path_dqn = f"{args.pretrain_path}/N{agentNum}/"
if mode == 'train':
    path = f"{args.save_path}/N{agentNum}/"
    if not os.path.exists(path):
        os.makedirs(path)
else:
    path = f"{args.load_path}/N{agentNum}/"
RL = HIST_Alg(a_dim, n_actions, s_dim, s_dim_ddpg, s_dim_dqn, a_bound, envSize, model_path_dqn, path, mode)

max_torque = env.max_torque
agentSize = env.agentSize
agentPositionArray = np.zeros((agentNum, 2))
agentPositionArray0 = np.zeros((agentNum, 2))
tarPositionArray = np.zeros((agentNum, 2))
tarPositionArray0 = np.zeros((agentNum, 2))
obstacleArray = np.zeros((obsNum, 2))
agentObstacleDisTS = np.ones((agentNum, detecDirNum))
step_total = 0
episode = 0
temp_rewardSum = 0
totalReward_list = []
meanReward_list = []
timeCostSum_temp = 0
meanTime_list = []
collision_num = 0
collision_obs_num = 0
collisionNum_list = []
success_num = 0
successNum_list = []
conflict_num = 0
timeStart = time.time()

for ep in range(ep_num):
    episode += 1
    # Generate obs, agent, tar
    locationArea = np.random.choice(36, agentNum * 2 + obsNum, replace=False)
    for i in range(obsNum):
        obstacleArray[i] = [locationArea[i] // 6 * 6 + 2.5, locationArea[i] % 6 * 6 + 2.5] + 1 * np.random.rand(2)
    for i in range(agentNum):
        tarPositionArray0[i] = [locationArea[obsNum + i] // 6 * 6 + 2, locationArea[obsNum + i] % 6 * 6 + 2] + 2 * np.random.rand(2)
        agentPositionArray0[i] = [locationArea[obsNum + agentNum + i] // 6 * 6 + 2, locationArea[obsNum + agentNum + i] % 6 * 6 + 2] + 2 * np.random.rand(2)
    sortTar_index = np.argsort(tarPositionArray0[:, 0])
    for i in range(agentNum):
        tarPositionArray[i, :] = tarPositionArray0[sortTar_index[i], :]
    sortAgent_index = np.argsort(agentPositionArray0[:, 0])
    for i in range(agentNum):
        agentPositionArray[i, :] = agentPositionArray0[sortAgent_index[i], :]
    obstacleSize = np.random.rand(obsNum) * 1.3 + 0.5

    observation = env.reset(agentPositionArray, tarPositionArray, obstacleArray, obstacleSize)
    ep_timeCost = 0
    agentObstacleDis = np.ones((agentNum, detecDirNum))
    agentExistObstacle_Target = np.zeros(agentNum)
    agentExistObstacle_Target_old = np.zeros(agentNum)
    tarAngle = np.zeros(agentNum)
    observationCA = np.hstack((observation[:, :2], agentObstacleDis))
    action = [int(val) for val in np.zeros(agentNum)]
    action_ = [int(val) for val in np.zeros(agentNum)]
    action_h = - np.ones(agentNum, dtype=np.int)
    move = np.zeros((agentNum, 2))
    agentDone = [int(val) for val in np.zeros(agentNum)]
    ep_reward = np.zeros(agentNum)
    observation_h = np.tile(observation[:, -(agentNum - 1) * 2:], historyStep)
    observation_h_temp = observation_h
    collision_cross = np.zeros(agentNum)

    for step in range(MAX_EP_STEPS):
        env.render()
        otherTarCoordi = np.zeros((agentNum, 2))
        action_ddpg = np.zeros(agentNum)
        for i in range(agentNum):
            agentExistObstacle_Target[i] = 0
            # Target selection
            if collision_cross[i] != 1:
                action[i] = RL.choose_action_dqn(np.hstack((observation[i], observation_h[i], observation[i, -2*(agentNum-1):])))
            tarAgentCoordi = (observation[i, 2*action[i]:2*action[i]+2]) * envSize
            tarAgentDis = np.linalg.norm(tarAgentCoordi)
            if tarAgentDis >= 1:
                for k in range(agentNum):
                    if k != action[i] and 1 < np.linalg.norm(observation[i, 2*k:2*k+2]) * envSize < 2.5:
                        otherTarCoordi[i] = -(observation[i, 2*k:2*k+2]) * envSize
                        break
                tarAgentDirCoordi = tarAgentCoordi / tarAgentDis
                # Detect obstacles
                agentExistObstacle_Target[i], agentObstacleDis[i, 0] = env.detect_obstacle(tarAgentDirCoordi, i, otherTarCoordi[i])
                if agentExistObstacle_Target[i] == 1:
                    if action[i] != action_[i] and step:
                        tarAngle[i] = math.asin(tarAgentCoordi[0] / tarAgentDis)
                        if tarAgentCoordi[1] >= 0:
                            if tarAgentCoordi[0] >= 0:
                                tarAngle[i] = np.pi - tarAngle[i]
                            if tarAgentCoordi[0] < 0:
                                tarAngle[i] = -np.pi - tarAngle[i]
                        for interval in range(3):   # Each +30° direction
                            tarAngleAround = tarAngle[i] + (interval+1) * np.pi / 6
                            tarAngleAround_PolarCoordi = np.array([np.sin(tarAngleAround), -np.cos(tarAngleAround)])
                            temp, agentObstacleDis[i, interval + 1] = env.detect_obstacle(tarAngleAround_PolarCoordi, i, otherTarCoordi[i])
                        for interval in range(3):  # Each -30° direction
                            tarAngleAround = tarAngle[i] - (interval+1) * np.pi / 6
                            tarAngleAround_PolarCoordi = np.array([np.sin(tarAngleAround), -np.cos(tarAngleAround)])
                            temp, agentObstacleDis[i, interval + 4] = env.detect_obstacle(tarAngleAround_PolarCoordi, i, otherTarCoordi[i])
                    observationCA[i] = np.hstack((observation[i, action[i]*2: action[i]*2+2], agentObstacleDis[i]))
                    action_ddpg[i] = RL.choose_action_ddpg(observationCA[i])
                    actionMove = tarAngle[i] + action_ddpg[i]
                    move[i, 0], move[i, 1] = env.step_ddpg(actionMove)
                else:  # No obstacles ahead
                    move[i, 0], move[i, 1] = env.step_dqn(action[i], observation[i], agentDone[i])
            else:  # Arrive
                move[i, 0], move[i, 1] = env.step_dqn(action[i], observation[i], agentDone[i])

        observation_, reward, done, agentDone, collision_cross, collision_agent, collision_obs, success, conflict, agentPositionArray = \
            env.move(move, agentExistObstacle_Target, otherTarCoordi, action, action_h, 0)

        action_h = action
        observation_h_temp[:, 2 * (agentNum - 1):] = observation_h[:, :-2 * (agentNum - 1)]
        observation_h_temp[:, :2 * (agentNum - 1)] = observation[:, -2 * (agentNum - 1):]
        step_total += 1
        if mode == 'train':  # Learn
            if RL.pointer_ddpg > RL.MEMORY_SIZE_ddpg and sum(agentExistObstacle_Target):
                RL.learn_ddpg()
                # print(f'ts: {action}, angle: {np.around(action_ddpg, decimals=2)}')
        ep_timeCost += 1
        agentExistObstacle_Target_old = agentExistObstacle_Target
        ep_reward += reward
        otherTarCoordi = np.zeros((agentNum, 2))

        # Store transitions
        for i in range(agentNum):
            if collision_cross[i] != 1:
                action_[i] = RL.choose_action_dqn(np.hstack((observation_[i], observation_h_temp[i], observation_[i, -2*(agentNum-1):])))
            else:
                action_[i] = action[i]
            agentNextDDPG = 0
            tarAgentCoordi = (observation_[i, 2 * action_[i]: 2 * action_[i] + 2]) * envSize
            tarAgentDis = np.linalg.norm(tarAgentCoordi)
            if tarAgentDis >= 1:
                for k in range(agentNum):
                    if k != action_[i] and 1 < np.linalg.norm(observation_[i, k*2:k*2+2]) * envSize < 2.5:
                        otherTarCoordi[i] = -(observation_[i, k*2:k*2+2]) * envSize
                        break
                tarAgentDirCoordi = tarAgentCoordi / tarAgentDis
                agentNextDDPG, agentObstacleDis[i, 0] = env.detect_obstacle(tarAgentDirCoordi, i, otherTarCoordi[i])
                if agentNextDDPG == 1:
                    tarAngle[i] = math.asin(tarAgentCoordi[0] / tarAgentDis)
                    if tarAgentCoordi[1] >= 0:
                        if tarAgentCoordi[0] >= 0:
                            tarAngle[i] = np.pi - tarAngle[i]
                        if tarAgentCoordi[0] < 0:
                            tarAngle[i] = -np.pi - tarAngle[i]
                    for interval in range(3):
                        tarAngleAround = tarAngle[i] + (interval+1) * np.pi / 6
                        tarAngleAround_PolarCoordi = np.array([np.sin(tarAngleAround), -np.cos(tarAngleAround)])
                        temp, agentObstacleDis[i, interval + 1] = env.detect_obstacle(tarAngleAround_PolarCoordi, i, otherTarCoordi[i])
                    for interval in range(3):
                        tarAngleAround = tarAngle[i] - (interval+1) * np.pi / 6
                        tarAngleAround_PolarCoordi = np.array([np.sin(tarAngleAround), -np.cos(tarAngleAround)])
                        temp, agentObstacleDis[i, interval + 4] = env.detect_obstacle(tarAngleAround_PolarCoordi, i, otherTarCoordi[i])
            observation_All = np.hstack((observation_[i], observation_h_temp[i], observation_[i, -2*(agentNum-1):], agentObstacleDis[i]))
            if mode == 'train' and agentExistObstacle_Target[i] == 1:
                RL.store_transition_ddpg(observationCA[i], action_ddpg[i], reward[i], observation_All, action_[i], done[i], agentNextDDPG)
        observation_h = observation_h_temp
        observation = observation_

        if sum(done) or step == MAX_EP_STEPS - 1:
            collision_num += collision_agent
            collision_obs_num += collision_obs
            success_num += success
            conflict_num += conflict
            temp_rewardSum += min(ep_reward)
            if mode == 'train':
                print(f'Ep: {episode}, R: {np.around(min(ep_reward), decimals=3)}')
            if success == 0:
                ep_timeCost = MAX_EP_STEPS
            timeCostSum_temp += ep_timeCost
            break
    if mode == 'train':
        if episode % 100 == 0:
            print(" ~~~~~~~  Statistics ~~~~~~~~  Ep ", episode)
            print(f"Success: {success_num}, Collision: {collision_num}")
            mean_reward = temp_rewardSum / 100
            meanReward_list.append(mean_reward)
            if success_num >= 3:
                mean_time = timeCostSum_temp / success_num
                meanTime_list.append(mean_time)
            temp_rewardSum, timeCostSum_temp = 0, 0
            collisionNum_list.append(collision_num)
            collision_num = 0
            collision_obs_num = 0
            successNum_list.append(success_num)
            success_num = 0
            conflict_num = 0
    elif episode == ep_num:
        print(f" ~~~~~~~  Statistical result Ep {episode} ~~~~~~~~")
        print(f"Success: {success_num}, Collision: {collision_num}, Normalized time: {np.around(timeCostSum_temp/ep_num/MAX_EP_STEPS, decimals=3)}")

if mode == 'train':
    RL.save_Parameters()
    np.savetxt(path+'meanReward.txt', meanReward_list)
print(f"Finished! Running time: {time.time() - timeStart}")
