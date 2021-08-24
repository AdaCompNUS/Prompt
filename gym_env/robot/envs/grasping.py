import json
import math
import time

import cv2
import numpy as np
import pybullet as p
from gym import spaces
from robot.envs.base import BaseEnv


class GraspingEnv(BaseEnv):

    def __init__(self, GUI=True):
        super().__init__()
        self.GUI = GUI
        self.action_space = spaces.Discrete(4)  # center of gripper and rotation (x,y,z,theta)

    def save_image(self, img_path, seq=0):
        if img_path is not None:
            img_path_tmp = img_path.replace(".png", f"{seq}.png")
            rgb, img = self.get_image([0.6, 0.6, 0.3], [0.4, 0.4, 0.01], pixelWidth_=640, pixelHeight_=480)
            rgb = (rgb * 255).astype(np.uint8)
            cv2.imwrite(img_path_tmp, rgb)

    def step(self, action: np.ndarray, img_path=None):

        #  movement
        obj_original_pos = p.getAABB(self.obj_bullet_id)
        prepare_pos = action[0:3]
        prepare_pos[2] += 0.15
        self.save_image(img_path, 0)
        self.gripper(open_gripper=True)
        self.save_image(img_path, 1)
        self.move(prepare_pos, action[3])
        self.save_image(img_path, 2)
        self.move(action[0:3], action[3])
        self.save_image(img_path, 3)

        # open/close gripper
        self.gripper(open_gripper=False)
        self.save_image(img_path, 4)

        self.move(prepare_pos, action[3])
        self.save_image(img_path, 5)

        # reward
        reward = 0
        episode_over = False
        box = p.getAABB(self.obj_bullet_id)
        if box[1][2] - obj_original_pos[1][2] > 0.05:
            reward = 1
            episode_over = True

        self.steps += 1
        obs = {'image': None}
        return obs, reward, episode_over, {}

    def reset(self):
        np.random.seed(int(time.time()))
        self.connect_pybullet(GUI=self.GUI)
        self.visual_ids = []

        self.steps = 0
        self.rest_joints()
        self.gripper(True)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return {'image': None}

    def visualize_grasp(self, args, action):
        with open(args.gripper_conf) as f:
            gripper_conf = json.load(f)

        width = gripper_conf['rest_gripper_dis'] - gripper_conf['halfEdge'][0] * 2
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.01, width, 0.002],
                                            rgbaColor=[0, 1, 0, 1])
        orn = p.getQuaternionFromEuler([0, -math.pi, action[3]])
        obj_id = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=-1,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=[action[0] + 0.2, action[1], action[2]],
                                   baseOrientation=orn,
                                   useMaximalCoordinates=True)
        self.visual_ids.append(obj_id)
