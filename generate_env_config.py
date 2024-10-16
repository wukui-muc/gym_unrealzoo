import re
import time
from unrealcv.api import UnrealCv_API
from unrealcv.launcher import RunUnreal
from unrealcv.util import measure_fps, parse_resolution
import argparse
import json
import copy
import numpy as np
'''
An example to show how to use the UnrealCV API to launch the game and run some functions
'''
class_name = {
    "player": "bp_character_C",
    "animal": "BP_animal_C",
    "drone": "BP_drone01_C",
    "car": "BP_BaseCar_C",
    "motorbike": "MotorBikes_C",
}

player_config = {
        "name": [],
        "cam_id": [],
        "class_name": [],
        "internal_nav": True,
        "scale": [1, 1, 1],
        "relative_location": [20, 0, 0],
        "relative_rotation": [0, 0, 0],
        "head_action_continuous": {
            "high": [15, 15, 15],
            "low":  [-15, -15, -15]
        },
        "head_action": [
            [0, 0, 0], [0, 30, 0], [0, -30, 0]],
        "animation_action": ["stand", "jump", "crouch"],
        "move_action": [
            [0, 100], [0, -100], [15, 50], [-15, 50], [30, 0], [-30, 0], [0, 0]
        ],
        "move_action_continuous": {
            "high": [30, 100],
            "low": [-30, -100]
        }
    }

animal_config = {
        "name": [],
        "cam_id": [],
        "class_name": [],
        "internal_nav": True,
        "scale": [1, 1, 1],
        "relative_location": [20, 0, 0],
        "relative_rotation": [0, 0, 0],
        "move_action": [
            [0, 200],
            [0, -200],
            [15, 100],
            [-15, 100],
            [30, 0],
            [-30, 0],
            [0, 0]
        ],
        "move_action_continuous": {
            "high": [30, 200],
            "low": [-30, -200]
        }
    }

drone_config = {
    "name": [],
    "cam_id": [],
    "class_name": [],
    "internal_nav": False,
    "scale": [0.1, 0.1, 0.1],
    "relative_location": [0, 0, 0],
    "relative_rotation": [0, 0, 0],
    "move_action": [
        [0.5, 0, 0, 0],
        [-0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, -0.5, 0, 0],
        [0, 0, 0.5, 0],
        [0, 0, -0.5, 0],
        [0, 0, 0, 1],
        [0, 0, 0, -1],
        [0, 0, 0, 0]
    ],
    "move_action_continuous": {
        "high": [1, 1, 1, 1],
        "low": [-1, -1, -1, -1]
    }
    }
car_config = {
    "name": [],
    "cam_id": [],
    "class_name": [],
    "internal_nav": True,
    "scale": [1, 1, 1],
    "relative_location": [0, 0,  0],
    "relative_rotation": [0, 0, 0],
    "move_action": [
        [ 1,  0],
        [ -0.3,  0],
        [ 0.5,  1],
        [ 0.5, -1],
        [ 0,  0]
    ],
    "move_action_continuous": {
        "high": [ 1,  1],
        "low":  [0, -1]
    }
}

motorbike_config = {
    "name": [],
    "cam_id": [],
    "class_name": [],
    "internal_nav": True,
    "scale": [1, 1, 1],
    "relative_location": [0, 0,  0],
    "relative_rotation": [0, 0, 0],
    "move_action": [
        [1,  0],
        [-0.3,  0],
        [0.5,  1],
        [0.5, -1],
        [0,  0]
    ],
    "move_action_continuous": {
        "high": [1,  1],
        "low":  [0, -1]
    }
}

agents = {
    "player": player_config,
    "animal": animal_config,
    "drone": drone_config,
    "car": car_config,
    "motorbike": motorbike_config
}

env_config = {
    "env_name": None,
    "env_bin": None,
    "env_map": None,
    "env_bin_win": None,
    "third_cam": {
        "cam_id": 0,
        "pitch": -90,
        "yaw": 0,
        "roll": 0,
        "height_top_view": 1500,
        "fov": 90
    },
    "height": 500,
    "interval": 1000,
    "agents": agents,
    "safe_start": [],
    "reset_area": [0, 0, 0, 0, 0, 0],
    "random_init": False,
    "env": {
        "interactive_door": []
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-bin', default='E:\\gym_unrealcv2.0\\gym-unrealcv\\gym_unrealcv\\envs\\UnrealEnv\\Collection_WinNoEditor\\Collection\\Binaries\\Win64\\Collection.exe', help='The path to the UE4Editor binary')
    parser.add_argument('--env-bin', default='E:\\gym_unrealcv2.0\\gym-unrealcv\\gym_unrealcv\\envs\\UnrealEnv\\SuburbNeighborhood_Day_WindowsNoEditor\\SuburbNeighborhood\\Binaries\\Win64\\SuburbNeighborhood.exe', help='The path to the UE4Editor binary')

    parser.add_argument('--env-map', default='Venice', help='The map to load')
    parser.add_argument('--use-docker', action='store_true', help='Run the game in a docker container')
    parser.add_argument('--resolution', '-res', default='640x480', help='The resolution in the unrealcv.ini file')
    parser.add_argument('--display', default=None, help='The display to use')
    parser.add_argument('--use-opengl', action='store_true', help='Use OpenGL for rendering')
    parser.add_argument('--offscreen', action='store_true', help='Use offscreen rendering')
    parser.add_argument('--nullrhi', action='store_true', help='Use the NullRHI')
    parser.add_argument('--show', action='store_true', help='show the get image result')
    parser.add_argument('--gpu-id', default=None, help='The GPU to use')
    args = parser.parse_args()
    env_bin = args.env_bin
    env_map = args.env_map
    if args.env_map == 'all':
        maps = ['Greek_Island', 'supermarket', 'Brass_Gardens', 'Brass_Palace', 'Brass_Streets',
                'EF_Gus', 'EF_Lewis_1', 'EF_Lewis_2', 'EF_Grounds', 'Eastern_Garden', 'Western_Garden', 'Colosseum_Desert',
                'Desert_ruins', 'SchoolGymDay', 'Venice', 'TrainStation', 'Stadium', 'IndustrialArea', 'ModularBuilding',
                'TemplePlaza', 'DowntownWest', 'TerrainDemo', 'InteriorDemo_NEW', 'AncientRuins', 'Grass_Hills', 'ChineseWaterTown_Ver1',
                'ContainerYard_Night', 'ContainerYard_Day', 'Old_Factory_01', 'racing_track', 'Watermills', 'WildWest',
                'SunsetMap', 'Hospital', 'Medieval_Castle', 'Real_Landscape', 'UndergroundParking', 'Demonstration_Castle',
                'Demonstration_Cave', 'Arctic', 'Medieval_Daytime', 'Medieval_Nighttime', 'ModularGothic_Day', 'ModularGothic_Night',
                'UltimateFarming', 'RuralAustralia_Example_01', 'RuralAustralia_Example_02', 'RuralAustralia_Example_03',
                'LV_Soul_Cave', 'Dungeon_Demo_00', 'SwimmingPool', 'DesertMap', 'RainMap', 'SnowMap', 'ModularVictorianCity_scene1',
                'SuburbNeighborhood_Day', 'SuburbNeighborhood_Night', 'ModularSciFiVillage', 'Storagehouse', 'OceanFloor',
                'ModularNeighborhood', 'ModularSciFiVillage', 'ModularSciFiSeason1', 'LowPolyMedievalInterior_1',
                'QA_Holding_Cells_A', 'ParkingLot','track_train'
                ]
        env_map = maps[0]
    else:
        maps = [env_map]
    print(len(maps))
    ue_binary = RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)
    env_ip, env_port = ue_binary.start(args.use_docker, parse_resolution(args.resolution), args.display, args.use_opengl, args.offscreen, args.nullrhi, str(args.gpu_id))
    unrealcv = UnrealCv_API(env_port, env_ip, parse_resolution(args.resolution), 'tcp')  # 'tcp' or 'unix', 'unix' is only for local machine in Linux
    # unrealcv.config_ue(parse_res(args.resolution))
    for env_map in maps:
        unrealcv.set_map(env_map)
        agents = {
            "player": copy.deepcopy(player_config),
            "animal": copy.deepcopy(animal_config),
            "drone": copy.deepcopy(drone_config),
            "car": copy.deepcopy(car_config),
            "motorbike": copy.deepcopy(motorbike_config)
        }
        env_config = {
            "env_name": None,
            "env_bin": None,
            "env_map": None,
            "env_bin_win": None,
            "third_cam": {
                "cam_id": 0,
                "pitch": -90,
                "yaw": 0,
                "roll": 0,
                "height_top_view": 1500,
                "fov": 90
            },
            "height": 500,
            "interval": 1000,
            "agents": agents,
            "safe_start": [],
            "reset_area": [0, 0, 0, 0, 0, 0],
            "random_init": False,
            "env": {
                "interactive_door": []
            }
        }

        env_config['env_name'] = env_map
        env_config['env_map'] = env_map
        env_config['env_bin'] = ue_binary.env_bin
        env_config['env_bin_win'] = ue_binary.env_bin.replace("/", "\\").replace("Linux", "Win") + ".exe"
        # time.sleep(1)
        cam_num = unrealcv.get_camera_num()
        start_pos_list = []
        cam_locs = []
        for i in range(cam_num):
            loc = unrealcv.get_cam_location(i)
            cam_locs.append(loc)
        # Test the API
        objects = unrealcv.get_objects()

        obj_locations = []
        obj_size = []
        obj_info = {}
        # print(objects)
        print(env_map, len(objects))
        env_config['obj_num'] = len(objects)
        for obj in objects:
            if 'RecastNavMesh' in obj:
                uclass = unrealcv.get_obj_uclass(obj)
                # bbox = unrealcv.get_obj_size(obj, box=True)
                bbox = unrealcv.get_obj_size(obj)
                bbox[0] = bbox[0]/100.0
                bbox[1] = bbox[1]/100.0
                bbox[2] = bbox[2]/100.0
                size = bbox[0] * bbox[1] * bbox[2]
                area = bbox[0] * bbox[1]
                print(obj, uclass, bbox, size, area)
                env_config['size'] = size
                env_config['area'] = area
                env_config['bbox'] = bbox
        # for obj in objects:
        #     if 'NavMesh' in obj or 'PostProcess' in obj or 'Capture' in obj or 'SkyLight' in obj or 'Light' in obj or 'DirectionalLight' in obj or 'ExponentialHeightFog' in obj or 'AtmosphericFog' in obj or 'ReflectionCapture' in obj or 'SphereReflectionCapture' in obj or 'PlanarReflection' in obj:
        #         continue
        #     uclass = unrealcv.get_obj_uclass(obj)
        #     bbox = unrealcv.get_obj_size(obj, box=True)
        #     size = bbox[0] * bbox[1] * bbox[2]
        #     location = unrealcv.get_obj_location(obj)
        #     if size == 0:
        #         continue
        #     obj_data = {}
        #     obj_data['size'] = size
        #     obj_data['bbox'] = bbox
        #     obj_data['location'] = unrealcv.get_obj_location(obj)
        #     obj_info[obj] = obj_data
            # obj_size.append(unrealcv.get_obj_size(obj, box=True))
            # obj_locations.append(unrealcv.get_obj_location(obj))
        # rint(obj_info, len(obj_info))

        def generate_nav_goal(player, radius_max,radius_min):
            cmd = f'vbp {player} generate_nav_goal {radius_max} {radius_min}'
            res = unrealcv.client.request(cmd)
            goal = unrealcv.decoder.string2vector(res)
            if len(goal) == 0:
                print(env_map)
                print(f'Failed to generate nav goal for {player}')
                return None
            return goal

        def match_cam_id(cam_locs, obj_name):
            obj_loc = unrealcv.get_obj_location(obj_name)
            dis_list = []
            for loc in cam_locs:
                distance = unrealcv.get_distance(loc, obj_loc, 3)
                dis_list.append(distance)
            cam_id = dis_list.index(min(dis_list))
            return cam_id
        for obj in objects:
            if re.match(re.compile(r'bp_character', re.I), obj) is not None:
                agents['player']['name'].append(obj)
                agents['player']['class_name'].append(class_name['player'])
                agents['player']['cam_id'].append(match_cam_id(cam_locs, obj))
                start_pos_list.append(unrealcv.get_obj_location(obj))
                for i in range(10):
                    # print(generate_nav_goal(obj, 1000))
                    goal_loc = generate_nav_goal(obj, 4000,500)
                    print(goal_loc)
                    if goal_loc is not None:
                        goal_loc[-1] += 50
                        start_pos_list.append(goal_loc)
            elif re.match(re.compile(r'bp_animal', re.I), obj) is not None:
                agents['animal']['name'].append(obj)
                agents['animal']['class_name'].append(class_name['animal'])
                agents['animal']['cam_id'].append(match_cam_id(cam_locs, obj))
                start_pos_list.append(unrealcv.get_obj_location(obj))
            elif re.match(re.compile(r'bp_drone', re.I), obj) is not None:
                agents['drone']['name'].append(obj)
                agents['drone']['cam_id'].append(match_cam_id(cam_locs, obj))
                agents['drone']['class_name'].append(class_name['drone'])
                # env_config['safe_start'].append(unrealcv.get_obj_location(obj))
            elif re.match(re.compile(r'bp_basecar', re.I), obj) is not None:
                agents['car']['name'].append(obj)
                agents['car']['cam_id'].append(match_cam_id(cam_locs, obj))
                agents['car']['class_name'].append(class_name['car'])
                start_pos_list.append(unrealcv.get_obj_location(obj))
            elif re.match(re.compile(r'sport', re.I), obj) is not None or re.match(re.compile(r'motorbike', re.I), obj) is not None:
                agents['motorbike']['name'].append(obj)
                agents['motorbike']['cam_id'].append(match_cam_id(cam_locs, obj))
                agents['motorbike']['class_name'].append(class_name['motorbike'])
                start_pos_list.append(unrealcv.get_obj_location(obj))
            elif re.match(re.compile(r'bp_door', re.I), obj) is not None or re.match(re.compile(r'animateddoor', re.I), obj) is not None:
                env_config['env']['interactive_door'].append(obj)

        agents = {k: v for k, v in agents.items() if len(v['name']) > 0}  # remove the agent category not in the scene
        env_config['agents'] = agents

        env_config['safe_start'] = start_pos_list
        cam_x = [cam_loc[0] for cam_loc in start_pos_list]
        cam_y = [cam_loc[1] for cam_loc in start_pos_list]
        cam_z = [cam_loc[2] for cam_loc in start_pos_list]
        env_config['height'] = max(cam_z)
        env_config['reset_area'] = [min(cam_x), max(cam_x), min(cam_y), max(cam_y), min(cam_z), max(cam_z)]
        env_config['third_cam']['height_top_view'] = env_config['height'] + 1000
        # print(env_config)
        import os
        # target_dir = 'gym_unrealcv/envs/setting/env_config'
        target_dir = './statistic'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(os.path.join(target_dir, f'{env_map}.json'), 'w') as json_file:
            json.dump(env_config, json_file, indent=4)


    unrealcv.client.disconnect()
    ue_binary.close()