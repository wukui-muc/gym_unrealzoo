import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
import os
# import torch
from gym_unrealcv.envs.tracking.baseline import PoseTracker
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
import random
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        self.action = self.action_space.sample()

    def act(self, observation, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            self.action = self.action_space.sample()
            self.count_steps = 0
        else:
            return self.action
        return self.action

    def reset(self):
        self.action = self.action_space.sample()
        self.count_steps = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealSearch-IndustrialAreaPoint-DiscreteMask-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=30, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
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
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False
    try:
        for i in range(episode_count):
            env.seed(i)
            ob = env.reset()
            count_step = 0
            t0 = time.time()
            while True:
                action = agent.act(ob, keep_steps=1)
                ob, reward, done, _ = env.step(action)
                cv2.imshow('mask',ob.astype(np.uint8))
                cv2.waitKey(1)
                count_step += 1
                print(count_step)
                if args.render:
                    img = env.render(mode='rgb_array')
                    #  img = img[..., ::-1]  # bgr->rgb
                    cv2.imshow('show', img)
                    cv2.waitKey(1)
                if done:
                    fps = count_step / (time.time() - t0)
                    print('Fps:' + str(fps))
                    break

        # Close the env and write monitor result info to disk
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()
