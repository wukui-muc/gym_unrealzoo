# from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
from unrealcv.api import UnrealCv_API
from gym_unrealcv.envs.utils import env_unreal, misc
import numpy as np
import time
from gym import spaces
import gym
import distutils.version
import re
from io import BytesIO
import PIL.Image
import cv2
import json
class Navigation(UnrealCv_API):
    def __init__(self, port=9000, ip='127.0.0.1', resolution=(160, 120), comm_mode='tcp'):
        super(Navigation, self).__init__(port=port, ip=ip, resolution=resolution, mode=comm_mode)

        self.img_color = np.zeros(1)
        self.img_depth = np.zeros(1)

        self.use_gym_10_api = distutils.version.LooseVersion(gym.__version__) >= distutils.version.LooseVersion('0.10.0')
        # self.client.request(f'vrun setres {64}x{64}w', -1)  # set resolution of the display window
        # self.client.request('DisableAllScreenMessages', -1)  # disable all screen messages
        # self.client.request('vrun sg.ShadowQuality 0', -1)  # set shadow quality to low
        # self.client.request('vrun sg.TextureQuality 0', -1)  # set texture quality to low
        # self.client.request('vrun sg.EffectsQuality 0', -1)  # set effects quality to low

    def init_mask_color(self, targets=None):
        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)
    def set_cam_illumination(self, cam_id,param):
        cmd = f'vset /camera/{cam_id}/illumination {param}'
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_cam_reflection(self, cam_id, param):
        cmd = f'vset /camera/{cam_id}/reflection {param}'
        res = None
        while res is None:
            res = self.client.request(cmd, -1)
    def set_move_bp(self, player, params, return_cmd=False):
        '''
        new move function, can adapt to different number of params
        2 params: [v_angle, v_linear], used for agents moving in plane, e.g. human, car, animal
        4 params: [v_ x, v_y, v_z, v_yaw], used for agents moving in 3D space, e.g. drone
        '''
        params_str = ' '.join([str(param) for param in params])
        cmd = f'vbp {player} set_move {params_str}'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    # functions for character actions
    def set_jump(self, player, return_cmd=False):
        cmd = f'vbp {player} set_jump'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def set_crouch(self, player, return_cmd=False):
        cmd = f'vbp {player} set_crouch'
        if return_cmd:
            return cmd
        res = None
        while res is None:
            res = self.client.request(cmd, -1)

    def get_hit(self, target):
        cmd = f'vbp {target} get_hit'
        res = None
        while res is None:
            res = self.client.request(cmd)
        data = json.loads(res)
        return int(data['Hit'])
    def get_observation(self, cam_id, observation_type, mode='direct'):
        if observation_type == 'Color':
            self.img_color = state = self.read_image(cam_id, 'lit', mode)
        elif observation_type == 'Mask':
            self.img_color = state = self.read_image(cam_id, 'object_mask', mode)
        elif observation_type == 'Depth':
            self.img_depth = state = self.get_depth(cam_id)
        elif observation_type == 'Rgbd': 
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_depth = self.get_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)
        elif observation_type == 'CG':
            self.img_color = self.read_image(cam_id, 'lit', mode)
            self.img_gray = self.img_color.mean(2)
            self.img_gray = np.expand_dims(self.img_gray, -1)
            state = np.concatenate((self.img_color, self.img_gray), axis=2)
        elif observation_type == 'Pose':
            state = self.get_pose() # fake pose
        return state

    def define_observation(self, cam_id, observation_type, mode='direct'):
        if observation_type == 'Pose' or cam_id < 0:
            observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float16) # TODO check the range and shape
        else:
            state = self.get_observation(cam_id, observation_type, mode)
            if observation_type == 'Color' or observation_type == 'CG' or observation_type == 'Mask':
                if self.use_gym_10_api:
                    observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)  # for gym>=0.10
                else:
                    observation_space = spaces.Box(low=0, high=255, shape=state.shape)
            elif observation_type == 'Depth':
                if self.use_gym_10_api:
                    observation_space = spaces.Box(low=0, high=100, shape=state.shape, dtype=np.float16)  # for gym>=0.10
                else:
                    observation_space = spaces.Box(low=0, high=100, shape=state.shape)
            elif observation_type == 'Rgbd':
                s_high = state
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                s_low = np.zeros(state.shape)
                if self.use_gym_10_api:
                    observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)  # for gym>=0.10
                else:
                    observation_space = spaces.Box(low=s_low, high=s_high)

        return observation_space

    def open_door(self):
        self.keyboard('RightMouseButton')
        time.sleep(2)
        self.keyboard('RightMouseButton')  # close the door

    def set_texture(self, target, color=(1, 1, 1), param=(0, 0, 0), picpath=None, tiling=1, e_num=0): #[r, g, b, meta, spec, rough, tiling, picpath]
        param = param / param.max()
        # color = color / color.max()
        cmd = 'vbp {target} set_mat {e_num} {r} {g} {b} {meta} {spec} {rough} {tiling} {picpath}'
        self.client.request(cmd.format(target=target, e_num=e_num, r=color[0], g=color[1], b=color[2],
                                       meta=param[0], spec=param[1], rough=param[2], tiling=tiling,
                                       picpath=picpath), -1)

    def set_light(self, target, direction, intensity, color): # param num out of range
        [roll, yaw, pitch] = direction
        color = color / color.max()
        [r, g, b] = color
        cmd = f'vbp {target} set_light {roll} {yaw} {pitch} {intensity} {r} {g} {b}'
        self.client.request(cmd, -1)

    def set_skylight(self, obj, color, intensity ): # param num out of range
        [r, g, b] = color
        cmd = f'vbp {obj} set_light {r} {g} {b} {intensity} '
        self.client.request(cmd, -1)

    def get_pose(self, cam_id, type='hard'):  # pose = [x, y, z, roll, yaw, pitch]
        if type == 'soft':
            pose = self.cam[cam_id]['location']
            pose.extend(self.cam[cam_id]['rotation'])
            return pose

        if type == 'hard':
            self.cam[cam_id]['location'] = self.get_cam_location(cam_id)
            self.cam[cam_id]['rotation'] = self.get_cam_rotation(cam_id)
            pose = self.cam[cam_id]['location'] + self.cam[cam_id]['rotation']
            return pose
    def read_image(self, cam_id, viewmode, mode='direct'):
            # cam_id:0 1 2 ...
            # viewmode:lit,  =normal, depth, object_mask
            # mode: direct, file
            res = None
            if mode == 'direct': # get image from unrealcv in png format
                cmd = f'vget /camera/{cam_id}/{viewmode} png'
                image = self.decode_png(self.client.request(cmd))

            elif mode == 'file': # save image to file and read it
                cmd = f'vget /camera/{cam_id}/{viewmode} {viewmode}{self.ip}.png'
                if self.docker:
                    img_dirs_docker = self.client.request(cmd)
                    img_dirs = self.envdir + img_dirs_docker[7:]
                else :
                    img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)
            elif mode == 'fast': # get image from unrealcv in bmp format
                cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
                image = self.decode_bmp(self.client.request(cmd))
            return image
    def decode_png(self, res):  # decode png image
        img = np.asarray(PIL.Image.open(BytesIO(res)))
        img = img[:, :, :-1]  # delete alpha channel
        img = img[:, :, ::-1]  # transpose channel order
        return img

    def decode_bmp(self, res, channel=4):  # decode bmp image
        img = np.fromstring(res, dtype=np.uint8)
        img = img[-self.resolution[1] * self.resolution[0] * channel:]
        img = img.reshape(self.resolution[1], self.resolution[0], channel)
        return img[:, :, :-1]  # delete alpha channel

    def decode_depth(self, res):  # decode depth image
        depth = np.fromstring(res, np.float32)
        depth = depth[-self.resolution[1] * self.resolution[0]:]
        depth = depth.reshape(self.resolution[1], self.resolution[0], 1)
        return depth

    def get_relative(self, pose0, pose1):  # pose0-centric
        delt_yaw = pose1[4] - pose0[4]
        angle = misc.get_direction(pose0, pose1)
        distance = self.get_distance(pose1, pose0, 3)
        # distance_norm = distance / self.exp_distance
        obs_vector = [np.sin(delt_yaw/180*np.pi), np.cos(delt_yaw/180*np.pi),
                      np.sin(angle/180*np.pi), np.cos(angle/180*np.pi),
                      distance]
        return obs_vector, distance, angle
    def get_pose_states(self, obj_pos):
        # get the relative pose of each agent and the absolute location and orientation of the agent
        pose_obs = []
        player_num = len(obj_pos)
        np.zeros((player_num, player_num, 2))
        relative_pose = np.zeros((player_num, player_num, 2))
        for j in range(player_num):
            vectors = []
            for i in range(player_num):
                obs, distance, direction = self.get_relative(obj_pos[j], obj_pos[i])
                yaw = obj_pos[j][4]/180*np.pi
                # rescale the absolute location and orientation
                abs_loc = [obj_pos[i][0], obj_pos[i][1],
                           obj_pos[i][2], np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
                relative_pose[j, i] = np.array([distance, direction])
            pose_obs.append(vectors)

        return np.array(pose_obs), relative_pose