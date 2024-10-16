import cv2
from pynput import keyboard
import argparse
import gym_unrealcv
import gym
from queue import Queue
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE

queue = Queue()
global keys
# use keyboard "i,j,k,l" to control tracking players
#  i: move forward
#  k: move backward
#  j: turn anticlockwise
#  l: turn clockwise
def on_press(key):
    try:
        if key.char == 'r':
            env.reset()
        elif key.char in ['i','j','k','l']:
            queue.put(key.char)
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-IndustrialArea-DiscreteColor-v0')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env = configUE.ConfigUEWrapper(env, offscreen=False)
    env = agents.NavAgents(env, mask_agent=False)
    env.seed(int(args.seed))
    obs = env.reset()

    keyboard_thread=keyboard.Listener(on_press=on_press,on_release=on_release)
    keyboard_thread.start()
    print('start press button')
    actions = [6]
    while keyboard_thread.is_alive():
        cv2.imshow('obs', obs[0])
        cv2.waitKey(1)
        if env.unwrapped.action_type =="Discrete":
            action_id = queue.get()
            # if action_id == '1':
            #     actions = [discrete_actions[0],discrete_actions[4]]
            # elif action_id == '2':
            #     actions = [discrete_actions[1],discrete_actions[4]]
            # elif action_id == '3':
            #     actions = [discrete_actions[3],discrete_actions[4]]
            # elif action_id == '4':
            #     actions = [discrete_actions[2],discrete_actions[4]]
            if action_id == 'i':
                # actions = [discrete_actions[4],discrete_actions[0]]
                actions=[0]
            elif action_id == 'j':
                # actions = [discrete_actions[4],discrete_actions[3]]
                actions=[5]
            elif action_id == 'k':
                # actions = [discrete_actions[4],discrete_actions[1]]
                actions=[1]
            elif action_id == 'l':
                # actions = [discrete_actions[4],discrete_actions[2]]
                actions=[4]

        obs, rewards, done, info = env.step(actions)


        # env.unwrapped.unrealcv.set_move_batch(env.env.player_list,actions)

    # Close the env and write monitor result info to disk
    env.close()