import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
import imageio
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
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-SuburbNeighborhood_Day-ContinuousMask-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
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

    env = augmentation.RandomPopulationWrapper(env, 8 , 8, random_target=False)
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(320,240))
    if args.nav_agent:
        env = agents.NavAgents(env, mask_agent=False)
    episode_count = 2
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('output_old.avi', fourcc, 15.0, (640, 480))

    images=[]
    try:
        for eps in range(1, episode_count):
            obs = env.reset()
            agents_num = len(env.action_space)
            agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]  # reset agents
            count_step = 0
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            for i in range(1,len(env.unwrapped.player_list)):
                env.unwrapped.unrealcv.nav_to_random(env.unwrapped.player_list[i], 500, 1)
            # env.unwrapped.unrealcv.nav_to_random(env.unwrapped.player_list[1], 500, 1)
            env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[1],[664,140,121])
            env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.player_list[1],[0,180,0])
            t0 = time.time()

            while True:
                # actions = [agents[i].act(obs[i]) for i in range(agents_num)]
                actions = [None for i in range(agents_num)]

                obs, rewards, done, info = env.step(actions)
                # obs_tmp=env.unwrapped.unrealcv.get_image(0, 'lit')
                obs_tmp = obs[0]
                obs_tmp=  cv2.cvtColor(obs_tmp, cv2.COLOR_RGB2BGR)  # Convert color space if necessary

                images.append(obs_tmp)



                C_rewards += rewards
                count_step += 1
                elapsed_time = time.time() - t0
                fps = count_step / elapsed_time

                obs_tmp = cv2.putText(obs_tmp, 'FPS: {:.2f}'.format(fps), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                     (255, 255, 255),
                                     1, cv2.LINE_AA)
                # video.write(obs[1])

                cv2.imshow('obs', obs_tmp)
                cv2.waitKey(1)
                if done:
                    imageio.mimsave('output_new_multiagent_Mask.gif', images,fps=30)
                    fps = count_step/(time.time() - t0)
                    Total_rewards += C_rewards[0]
                    print ('Fps:' + str(fps), 'R:'+str(C_rewards), 'R_ave:'+str(Total_rewards/eps))
                    break
        video.release()
        # Close the env and write monitor result info to disk
        print('Finished')
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()


