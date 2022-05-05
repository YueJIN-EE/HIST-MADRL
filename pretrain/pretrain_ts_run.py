from pretrain_ts_env import ENV
from pretrain_ts_alg import SMADQN
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def run(mode, ep_num, save_path):
    step_total = 0
    episode = 0
    rewardSum_temp = 0
    meanReward_list = []
    timeCostSum_temp = 0
    meanTime_list = []
    num_success = 0
    num_conflict = 0
    timeStart = time.time()
    for ep in range(ep_num):
        episode += 1
        while True:  # Generate initial positions of targets and agents
            agentPositionArray0 = np.ones((agentNum, 2)) * envSize
            agentPositionArray = np.ones((agentNum, 2)) * envSize
            tarPositionArray0 = np.ones((agentNum, 2)) * envSize
            tarPositionArray = np.ones((agentNum, 2))
            for i in range(agentNum):
                agentPositionArray0[i] = np.multiply(agentPositionArray0[i], np.random.rand(2))
                tarPositionArray0[i] = np.multiply(tarPositionArray0[i], np.random.rand(2))
            dis_aa_ok = 0
            dis_ff_ok = 0
            for i in range(agentNum):
                for j in range(agentNum):
                    if j > i:
                        dis_aa_ok += np.linalg.norm(agentPositionArray0[i] - agentPositionArray0[j]) > 3
                        dis_ff_ok += np.linalg.norm(tarPositionArray0[i] - tarPositionArray0[j]) > 3
            if dis_aa_ok == agentNum * (agentNum - 1)/2 and dis_ff_ok == agentNum * (agentNum - 1)/2:
                break
        sortTar_index = np.argsort(tarPositionArray0[:, 0])
        for i in range(agentNum):
            tarPositionArray[i, :] = tarPositionArray0[sortTar_index[i], :]
        sortAgent_index = np.argsort(agentPositionArray0[:, 0])
        for i in range(agentNum):
            agentPositionArray[i, :] = agentPositionArray0[sortAgent_index[i], :]

        observation = env.reset(agentPositionArray, tarPositionArray)
        agentDone = [int(val) for val in np.zeros(agentNum)]
        action = [int(val) for val in np.zeros(agentNum)]
        action_ = [int(val) for val in np.zeros(agentNum)]
        move = np.zeros((agentNum, 2))
        ep_reward = np.zeros(agentNum)
        ep_timeCost = 0
        done_collision = np.zeros(agentNum)
        observation_h = np.tile(observation[:, -(agentNum-1)*2:], historyStep)
        observation_h_temp = observation_h

        for i in range(agentNum):
            action_[i] = RL.choose_action(np.hstack((observation[i], observation_h[i], observation[i, -2*(agentNum-1):])))
        for step in range(MAX_EP_STEPS):
            env.render()
            for i in range(agentNum):
                if done_collision[i] != 1:
                    action[i] = action_[i]
                move[i, 0], move[i, 1] = env.step(action[i], observation[i], agentDone[i])
            observation_, reward, done, agentDone, done_collision, done_collision_true, success, conflict = env.move(move, action)
            observation_h_temp[:, 2*(agentNum-1):] = observation_h[:, :-2*(agentNum-1)]
            observation_h_temp[:, :2*(agentNum-1)] = observation[:, -2*(agentNum-1):]
            for i in range(agentNum):
                action_[i] = RL.choose_action(np.hstack((observation_[i], observation_h_temp[i], observation_[i, -2*(agentNum-1):])))
                observation_All = np.hstack((np.hstack((observation_[i], observation_h_temp[i], observation_[i, -2*(agentNum-1):]))))
                if mode == 'train':  # Store transitions
                    RL.store_transition(np.hstack((observation[i], observation[i, -2*(agentNum-1):], observation_[i, -2*(agentNum-1):])), action[i], reward[i], action_[i], done, observation_All)
            if mode == 'train':  # Learn
                if (step_total > 200) and (step_total % 5 == 0):
                    RL.learn()
            observation_h = observation_h_temp   
            observation = observation_
            if ep:
                ep_reward += reward  
            step_total += 1
            ep_timeCost += 1
            if done or step == MAX_EP_STEPS - 1:
                rewardSum_temp += min(ep_reward)
                num_success += success
                num_conflict += conflict
                if success == 0:
                    ep_timeCost = MAX_EP_STEPS
                timeCostSum_temp += ep_timeCost
                break
        if mode == 'train':
            if episode % 100 == 0:
                print(f"~~~~~~~  Episode {episode} ~~~~~~~~")
                print(f"Success: {num_success}, Conflict: {num_conflict}")
                mean_reward = rewardSum_temp / 100
                meanReward_list.append(mean_reward)
                print('reward', np.around(mean_reward, decimals=2))
                mean_time = timeCostSum_temp/100
                meanTime_list.append(mean_time)
                rewardSum_temp, timeCostSum_temp = 0, 0
                num_success = 0
                num_conflict = 0
        elif episode == ep_num:
            print(f" ~~~~~~~  Statistical result Ep {episode} ~~~~~~~~")
            print(f"Success: {num_success}, Conflict: {num_conflict}")
    if mode == 'train':
        RL.save_parameters()
        np.savetxt(save_path+'meanReward.txt', meanReward_list)
        np.savetxt(save_path+'timeCost.txt', meanTime_list)
        print("Running time: ", time.time() - timeStart)
        plt.plot(np.arange(len(meanReward_list))*100, meanReward_list, 'b.-')
        plt.ylabel('Average episode reward')
        plt.xlabel('Number of training episodes')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_num", type=int, default=0)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--save_path", type=str, default='./result')
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
    envSize = env.ENV_H
    MAX_EP_STEPS = envSize * 2  # Maximum episode length
    envSize -= 0.99
    historyStep = env.historyStep
    print(f'N={env.agentNum}')
    save_path = f"{args.save_path}/N{env.agentNum}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path
    RL = SMADQN(env.n_actions, env.n_features, args.mode,
                learning_rate=0.01,
                reward_decay=1,
                replace_target_iter=300,
                memory_size=1500,
                batch_size=32,
                model_path=save_path
                )
    env.after(100, run(args.mode, ep_num, save_path))
    env.mainloop()
