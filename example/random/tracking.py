import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
from gym_unrealcv.envs.tracking.baseline import PoseTracker
import random
import os
# import torch
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import math
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # return 2
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-IndustrialArea-ContinuousColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=30,
                        help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true',
                        help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=100, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env = configUE.ConfigUEWrapper(env, offscreen=False)
    env = agents.NavAgents(env, mask_agent=False)
    episode_count = 100
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))

    try:
        for eps in range(0, episode_count):
            obs = env.reset()
            agents_num = len(env.action_space)
            tracker_id = env.unwrapped.tracker_id
            target_id = env.unwrapped.target_id
            tracker = PoseTracker(env.action_space[0], 200,0)
            tracker_random = RandomAgent(env.action_space[0])
            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            print('eps:', eps, 'agents_num:', agents_num)
            while True:
                obj_poses = env.unwrapped.obj_poses
                actions = [tracker.act(obj_poses[tracker_id], obj_poses[target_id])]
                obs, rewards, done, info= env.step(actions)
                C_rewards += rewards
                count_step += 1

                #visualize first-person observation
                cv2.imshow('show', obs[0])
                cv2.waitKey(1)
                if done:
                    fps = count_step / (time.time() - t0)
                    Total_rewards += C_rewards[0]
                    print('Fps:' + str(fps), 'R:' + str(C_rewards), 'R_ave:' + str(Total_rewards / eps),'Tracking Length:',count_step)
                    count_step=0
                    break

        # Close the env and write monitor result info to disk
        print('Finished')
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()