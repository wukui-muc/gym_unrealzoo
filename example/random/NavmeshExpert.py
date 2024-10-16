import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
import os
import torch
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
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-Demo_Roof-ContinuousMask-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=30, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
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
    episode_count = 5000
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))
    # goal_list = [[-30,150],[-15,150],[0,150],[15,150],[30,150],
    #         [-30,200],[-15,200],[0,200],[15,200],[30,200],
    #         [-30,250],[-15,250],[0,250],[15,250],[30,250],
    #         [-30,300],[-15,300],[0,300],[15,300],[30,300],
    #         [-30,350],[-15,350],[0,350],[15,350],[30,350]]
    # goal_list = [[0,200],[0,350],[0, 450],[-25,350],[25,350]]
    bbox_lsit=['bbox_0_250.png','bbox_0_350.png','bbox_0_450.png','bbox_goal-25_350.png','bbox_25_350.png']
    first_dimension = list(range(-25, 26,3))  # from -25 to 25
    second_dimension = list(range(200, 451,10))  # from 200 to 450

    # Create a list of tuples representing all combinations of the two dimensions
    goal_list = [(x, y) for x in first_dimension for y in second_dimension]
    try:
        for eps in range(1, episode_count):
            image = []
            action = []
            reward = []
            info_list =[]
            goa_id = random.randint(0, len(goal_list)-1)
            # goa_id = 5

            env.unwrapped.reward_params['exp_distance'] = goal_list[goa_id][1]
            env.unwrapped.reward_params['exp_angle'] = goal_list[goa_id][0]
            # env.unwrapped.reward_params['exp_distance'] = 200
            # env.unwrapped.reward_params['exp_angle'] = 0

            obs = env.reset()
            agents_num = len(env.action_space)
            # agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]  # reset agents
            # tracker = PoseTracker(env.action_space[0], env.unwrapped.reward_params['exp_distance'], env.unwrapped.reward_params['exp_angle'] )  # TODO support multi trackers
            tracker = PoseTracker(env.action_space[0], 200, 0)

            tracker_random = RandomAgent(env.action_space[0])
            target_random = RandomAgent(env.action_space[0])
            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            actions=[[0,0],[0,0]]
            goal = []
            flag=0
            dire_cnt=-30
            dis_cnt = 150

            #     tracker.expected_distance = int(np.random.randint(0, 30))+goal_list[goa_id][1]
            #     tracker.expected_angle = int(np.random.randint(0, 5))+goal_list[goa_id][0]
            # tracker.expected_distance = random.randint(200, 600)
            # tracker.expected_angle = random.randint(-25, 25)
            # tracker.expected_distance = goal_list[goa_id][1]
            # tracker.expected_angle = goal_list[goa_id][0]
            tracker.expected_distance = 600
            tracker.expected_angle = 0
            env.unwrapped.reward_params['exp_distance'] = tracker.expected_distance
            env.unwrapped.reward_params['exp_angle'] = tracker.expected_angle
            best_reward = -10
            while True:
                # actions = [agents[i].act(obs[i]) for i in range(agents_num)]
                # obs, rewards, done, info = env.step(actions)
                obs, rewards, done, info = env.step(actions)
                cv2.imshow('show',obs[0].astype(np.uint8))
                # if env.unwrapped.unrealcv.get_hit(env.unwrapped.player_list[env.unwrapped.tracker_id]):
                # if info['metrics']['target_viewed'] == 0 or env.unwrapped.unrealcv.get_hit(env.unwrapped.player_list[env.unwrapped.tracker_id]):#while tracker lost target view or get stuck (speed<5)
                #     print('lose target')
                #     target_pose = env.unwrapped.unrealcv.get_obj_location(
                #         env.unwrapped.player_list[env.unwrapped.target_id])
                #     # if info['Distance'] > 300:
                #     env.unwrapped.unrealcv.nav_to_goal_bypath(env.unwrapped.player_list[env.unwrapped.tracker_id],
                #                                               target_pose)
                #     tracker_speed = env.unwrapped.unrealcv.get_speed(
                #         env.unwrapped.player_list[env.unwrapped.tracker_id])
                #     tracker_angle = env.unwrapped.unrealcv.get_angle(
                #         env.unwrapped.player_list[env.unwrapped.tracker_id])
                #     actions = [[None,None]]
                # else:
                target_pose = env.unwrapped.unrealcv.get_obj_pose(
                    env.unwrapped.player_list[env.unwrapped.target_id])
                tracker_pose=env.unwrapped.unrealcv.get_obj_pose(
                    env.unwrapped.player_list[env.unwrapped.tracker_id])

                # std = 20
                # if count_step % 60 ==0:
                #     tracker.expected_distance = random.randint(200, 450)
                #     tracker.expected_angle = random.randint(-25, 25)
                # #     tracker.expected_distance = int(np.random.randint(0, 30))+goal_list[goa_id][1]
                # #     tracker.expected_angle = int(np.random.randint(0, 5))+goal_list[goa_id][0]
                #     env.unwrapped.reward_params['exp_distance'] = tracker.expected_distance
                #     env.unwrapped.reward_params['exp_angle'] = tracker.expected_angle
                print(env.unwrapped.reward_params['exp_angle'],env.unwrapped.reward_params['exp_distance'],rewards[0])
                if rewards[0]>=best_reward:
                    best_reward = rewards[0]
                    # gray = cv2.cvtColor(obs[0], cv2.COLOR_BGR2GRAY)
                    # _, goal_bbox = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
                    goal_bbox = obs[0]
                    cv2.imshow('mask', goal_bbox)
                # actions = [np.array(tracker.act(tracker_pose, target_pose)),target_random.act(obs,keep_steps=200)]
                actions = [np.array(tracker.act(tracker_pose, target_pose))]

                flag -= 1
                if random.random() < 0.1 * (400/tracker.expected_distance) or flag > 0:
                    actions[0] = tracker_random.act(obs,keep_steps=2)
                    # actions[0][0]= np.clip(actions[0][0],-30,30)
                    # smooth action
                    if flag <= 0:
                        action_tmp = actions[0]
                        flag = random.randint(2, 5)
                    else:
                        actions[0] = action_tmp
                # print('actions:',actions[0])



                image.append(obs[env.unwrapped.tracker_id])
                action.append([actions[0]])
                reward.append(rewards)
                info_list.append(info)
                goal.append([ env.unwrapped.reward_params['exp_angle'], env.unwrapped.reward_params['exp_distance']])
                # goal.append([0, 200])
                C_rewards += rewards
                count_step += 1
                if args.render:
                    img = env.render(mode='rgb_array')

                    #  img = img[..., ::-1]  # bgr->rgb
                    cv2.imshow('show', img.astype(np.uint8))
                    # cv2.imshow('depth', 600/(obs[0][:, :, -1]))
                cv2.imshow('rgb', (obs[0][:, :, 0:3].astype(np.uint8)))

                cv2.waitKey(1)
                if done:
                    fps = count_step/(time.time() - t0)
                    Total_rewards += C_rewards[0]
                    print ('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/eps))
                    dict = {
                        'action': action,
                        'image': image,
                        'reward': reward,
                        'info': info_list,
                        'goal': goal,
                        'goal_bbox':goal_bbox
                    }
                    #
                    # # video.release()
                    save_dir = os.path.join(
                        'E:\\FlexibleRoom_Continuous_dataset\\Continuous_goal_condition_mask_v0',
                        'Continuous_goal_condition_mask_v0' + "%04d" % int(eps) + "-%03d" % count_step + '.pt')
                    # save_dir = os.path.join(
                    # 'F:\\FlexibleRoom_Continuous_dataset\\imperfect_v1_GT_Withdisori\\',
                    # 'imperfect_v1_GT_Withdisori_' + "%04d" % int(eps) + "-%03d" % count_step + '.pt')
                    # if count_step==350:
                    #     torch.save(dict, save_dir)
                    break

        # Close the env and write monitor result info to disk
        print('Finished')
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()


