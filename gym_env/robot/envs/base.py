import math
import os
import random
import time
from math import tan
from pathlib import Path
import cv2
import gym
import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn.functional as F
from gym import spaces

default_height = 0.03
cameraUp = [0, 0, 1]
pixelWidth = 256
pixelHeight = 192
aspect = pixelWidth / pixelHeight
nearPlane = 0.01
farPlane = 1000
fov = 90


class BaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    env = None

    def __init__(self):
        abs_path = str(Path(__file__).absolute()).replace("/base.py", "")
        self.data_path = f"{abs_path}/../objects_demo"
        obj_list = os.listdir(self.data_path)
        self.obj_list = []
        for obj_name in obj_list:
            if "_vhacd" not in obj_name:
                self.obj_list.append(obj_name)
        self.obj_list.sort()

        self.low_state = np.float32(np.ones((pixelWidth, pixelHeight, 3)) * 0)
        self.high_state = np.float32(np.ones((pixelWidth, pixelHeight, 3)) * 255)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.uint8)

        self.kukaEndEffectorIndex = 11
        self.numJoints = 9
        self.max_step_per_action = 15000
        self.max_steps = 50

        self.kuka_id = None
        self.obj_bullet_id = None
        self.convex_obj_id = None
        # lower limits for null space
        self.ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, 0, -2.8973, -5, -5, 0, 0, -5]
        # upper limits for null space
        self.ul = [2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 2, 2.8973, 5, 5, 0.04, 0.04, 5]
        # joint ranges for null space
        self.jr = np.array(self.ul) - np.array(self.ll)
        # restposes for null space
        self.rp = [0, 0, 0, -1.5, 0, 0, 0, 0, 0, 0, 0, 0]

        self.steps = 0

    def connect_pybullet(self, GUI=False):
        option_string = '--width={} --height={}'.format(pixelWidth, pixelHeight)

        if not p.isConnected():
            p.connect(p.GUI if GUI else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
            print(pybullet_data.getDataPath())
            if GUI:
                p.resetDebugVisualizerCamera(1.5, 150, -30, (0.1, 0.1, 0))
                # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.p = p
        self.load_obj()
        self.kuka_id = p.loadURDF(f"{self.data_path}/../robot_model/panda.urdf")

    def move(self, pos, angle=0):
        steps = 0
        orn = p.getQuaternionFromEuler([0, -math.pi, angle])
        while steps < self.max_step_per_action:
            p.stepSimulation()
            joint_poses = p.calculateInverseKinematics(self.kuka_id, self.kukaEndEffectorIndex, pos, orn)
            for i, pose in enumerate(joint_poses):
                p.setJointMotorControl2(bodyIndex=self.kuka_id,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=pose,
                                        targetVelocity=0,
                                        force=200,
                                        positionGain=0.001,
                                        velocityGain=1)

            steps += 1
            curr_time = time.time() * 1000

            ls = p.getLinkState(self.kuka_id, self.kukaEndEffectorIndex)
            curr_pos = np.array(ls[0])
            target_pos = np.array(pos)
            distance = np.linalg.norm(target_pos - curr_pos)

            ori_dis = np.linalg.norm(np.array(ls[1]) - np.array(orn))

            if distance < 0.002 and ori_dis < 0.005:
                break
        return

    def gripper(self, open_gripper=False):
        target_pose = 0.04 if open_gripper else 0

        for i in range(self.max_step_per_action):
            p.setJointMotorControl2(bodyIndex=self.kuka_id,
                                    jointIndex=10,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_pose,
                                    targetVelocity=0,
                                    force=100,
                                    positionGain=0.01,
                                    velocityGain=1)
            p.setJointMotorControl2(bodyIndex=self.kuka_id,
                                    jointIndex=9,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_pose,
                                    targetVelocity=0,
                                    force=100,
                                    positionGain=0.01,
                                    velocityGain=1)
            p.stepSimulation()

    def rest_joints(self):
        rp = np.zeros(12)
        rp[3] = -1
        rp[5] = 1.8
        for i in range(200):
            for joint_i, joint_val in enumerate(rp):
                p.setJointMotorControl2(bodyIndex=self.kuka_id,
                                        jointIndex=joint_i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_val,
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=0.03,
                                        velocityGain=1)
            p.stepSimulation()

    def load_obj(self):
        if self.obj_bullet_id is not None:
            p.removeBody(self.obj_bullet_id)
        if self.convex_obj_id is not None:
            p.removeBody(self.convex_obj_id)

        obj_pos = np.random.random(2) * 0.01 + 0.4
        obj_pos = np.concatenate((obj_pos, [default_height]))
        obj_ori = p.getQuaternionFromEuler(np.random.random(3) * 360)

        obj_id = np.random.randint(0, len(self.obj_list) - 1)
        obj_path = f"{self.data_path}/{self.obj_list[obj_id]}"
        print("obj_id", obj_id)
        self.obj_id = obj_id

        vhacd_obj_path = obj_path.replace(".obj", "_vhacd.obj")
        if not os.path.exists(vhacd_obj_path):
            p.vhacd(obj_path, vhacd_obj_path, "vhacd_log.txt", alpha=0.04, resolution=100000)

        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=obj_path, meshScale=[1.5, 1.5, 1.5])
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=vhacd_obj_path,
                                                  meshScale=[1.5, 1.5, 1.5])
        self.obj_bullet_id = p.createMultiBody(baseMass=1,
                                               baseCollisionShapeIndex=collisionShapeId,
                                               baseVisualShapeIndex=visualShapeId,
                                               useMaximalCoordinates=True,
                                               basePosition=obj_pos,
                                               baseOrientation=obj_ori)
        # p.changeDynamics(self.obj_bullet_id, -1, lateralFriction=0.95)

        # convex_obj_pos = obj_pos + np.array([0.2, 0.2, 0])
        # self.convex_obj_id = p.createMultiBody(baseMass=1,
        #                                        baseCollisionShapeIndex=collisionShapeId,
        #                                        baseVisualShapeIndex=-1,
        #                                        useMaximalCoordinates=True,
        #                                        basePosition=convex_obj_pos,
        #                                        baseOrientation=obj_ori)

        p.changeVisualShape(self.obj_bullet_id, -1, rgbaColor=[1, 0.1, 0.1, 1])

        """
        possibility check
        objects may roll over away
        """
        # stabilize obj
        for i in range(self.max_step_per_action):
            p.stepSimulation()

        box = p.getAABB(self.obj_bullet_id)
        obj_pos = (np.array(box[1]) + np.array(box[0])) / 2
        init_obj_pos = obj_pos[:2]

        # stabilize obj
        for i in range(self.max_step_per_action):
            p.stepSimulation()

        box = np.array(p.getAABB(self.obj_bullet_id))
        obj_pos = (box[1] + box[0]) / 2

        # make sure object is not too big
        diagnal = np.linalg.norm(box[1] - box[0])
        if diagnal > 0.4:
            return self.load_obj()

        # check obj has min height
        if box[1][2] <= 0.012:
            return self.load_obj()

        # check object is reachable
        if 0.6 < obj_pos[0] or obj_pos[0] < 0.2 or 0.6 < obj_pos[1] or obj_pos[1] < 0.2:
            return self.load_obj()

        # check obj has moved
        if np.linalg.norm(obj_pos[:2] - init_obj_pos) > 0.001:
            return self.load_obj()

        base, ori = p.getBasePositionAndOrientation(self.obj_bullet_id)
        self.base_ori = p.getEulerFromQuaternion(ori)

    def get_image(self, camera_pos, cam_target_pos, pixelWidth_=None, pixelHeight_=None):
        light_color = [1, 1, 1]
        light_distance = 1.5

        pixelHeight_ = pixelHeight if pixelHeight_ is None else pixelHeight_
        pixelWidth_ = pixelWidth if pixelWidth_ is None else pixelWidth_

        view_matrix = p.computeViewMatrix(camera_pos, cam_target_pos, cameraUp)
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        img = p.getCameraImage(pixelWidth_, pixelHeight_, view_matrix, projection_matrix,
                               shadow=1,
                               lightColor=light_color,
                               lightDistance=light_distance,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb = np.array(img[2]).reshape(pixelHeight_, pixelWidth_, 4)[:, :, :3].astype(np.float32) / 255
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        return rgb, img

    def calc_extrinsic(self, camTargetPos, cameraPos, cameraUp):
        C = torch.tensor(cameraPos).float()
        p = torch.tensor(camTargetPos).float()
        u = torch.tensor(cameraUp).float()

        L = p - C
        L = F.normalize(L, dim=0, p=2)

        s = torch.cross(L, u)
        s = F.normalize(s, dim=0, p=2)
        u2 = torch.cross(s, L)
        r_mat = torch.tensor([
            [s[0], s[1], s[2]],
            [u2[0], u2[1], u2[2]],
            [-L[0], -L[1], -L[2]],
        ])

        m_r = torch.zeros((4, 4))
        m_r[:3, :3] = r_mat
        m_r[3, 3] = 1

        t = - torch.matmul(r_mat, C)
        m_t = torch.zeros((3, 4))
        m_t[:3, :3] = torch.eye(3)
        m_t[:3, 3] = t

        return m_r.numpy(), m_t.numpy()

    def calc_intrinsic(self):
        intrinsic = np.zeros((3, 3))
        intrinsic[0, 0] = -pixelWidth / (2 * tan(fov / 2 / 180 * math.pi)) / aspect
        intrinsic[1, 1] = pixelHeight / (2 * tan(fov / 2 / 180 * math.pi))
        intrinsic[0, 2] = pixelWidth / 2
        intrinsic[1, 2] = pixelHeight / 2
        intrinsic[0, 1] = 0  # skew
        intrinsic[2, 2] = 1

        return intrinsic

    def projection_unit_test(self, m_r, m_t, intrinsic, rgb, x, y, z):
        intrinsic = torch.tensor(intrinsic).float()
        dotPos = torch.tensor([x, y, z + 0.05]).view(1, 3).float()
        calc_projection_const = torch.ones((1, dotPos.size(0)))
        dotPos = dotPos.permute((1, 0))
        dotPos = torch.cat((dotPos, calc_projection_const.clone()), dim=0)

        m2 = torch.matmul(m_r, dotPos)
        m2 = torch.matmul(m_t, m2)
        new_p = torch.matmul(intrinsic, m2)
        ratio = torch.ones((1, dotPos.size(1))) / new_p[2, :]
        new_p = new_p * ratio
        new_p = new_p[:2, :]
        new_p = new_p.permute((1, 0))

        new_p = new_p.squeeze().numpy()

        rgb = cv2.circle(cv2.UMat(rgb), (int(new_p[0]), int(new_p[1])), 5, (255, 0, 0), 2)
        cv2.imshow("rgb", rgb)
        cv2.waitKey(0)

        return new_p

    def show_point_cloud(self, points, rgbaColor=[0, 0.7, 0.7, 1]):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=rgbaColor, radius=0.001)
        for point in points:
            obj_id = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=-1,
                                       baseVisualShapeIndex=visualShapeId,
                                       basePosition=point[0:3],
                                       useMaximalCoordinates=True)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.removeAllUserDebugItems()

        return obj_id

    def get_video(self, step=4):
        r = 0.18
        base, ori = p.getBasePositionAndOrientation(self.obj_bullet_id)
        x, y, z = base

        m_r_list = []
        m_t_list = []
        img_list = []
        rgb_list = []
        intrinsic = self.calc_intrinsic()
        num_imgs = 100 / step * 2

        for theta in range(0, 90, step):
            x_ = x + r * np.cos(np.radians(theta + 0.01))
            z_ = z + r * np.sin(np.radians(theta + 0.01))
            m_r, m_t = self.calc_extrinsic([x, y, z], [x_, y, z_], [0, 0, 1])
            m_r_list.append(m_r)
            m_t_list.append(m_t)

            rgb, img = self.get_image([x_, y, z_], [x, y, z])
            seg_img = np.array(img[4]).reshape(pixelHeight, pixelWidth)
            seg_img = (seg_img == 1).astype(float)
            img_list.append(seg_img)
            rgb_list.append(rgb)
            print(f"collecting multi-view images: {len(img_list) / num_imgs * 100:.1f}% complete")

        for theta in range(0, 90, step):
            y_ = y + r * np.cos(np.radians(theta + 0.01))
            z_ = z + r * np.sin(np.radians(theta + 0.01))
            m_r, m_t = self.calc_extrinsic([x, y, z], [x, y_, z_], [0, 0, 1])
            m_r_list.append(m_r)
            m_t_list.append(m_t)

            rgb, img = self.get_image([x, y_, z_], [x, y, z])
            seg_img = np.array(img[4]).reshape(pixelHeight, pixelWidth)
            seg_img = (seg_img == 1).astype(float)
            img_list.append(seg_img)
            rgb_list.append(rgb)
            print(f"collecting multi-view images: {len(img_list) / num_imgs * 100:.1f}% complete")

        return m_r_list, m_t_list, img_list, rgb_list, intrinsic

    def get_video_top_only(self):
        # used for rough position estimation
        base, ori = p.getBasePositionAndOrientation(self.obj_bullet_id)
        x, y, z = base

        m_r_list = []
        m_t_list = []
        img_list = []
        rgb_list = []
        intrinsic = self.calc_intrinsic()

        for step in range(-10, 10, 6):
            y_ = y + step * 0.01
            m_r, m_t = self.calc_extrinsic(camTargetPos=[x, y_ - 0.00001, 0], cameraPos=[x, y_, 0.3],
                                           cameraUp=[0, 0, 1])
            m_r_list.append(m_r)
            m_t_list.append(m_t)

            rgb, img = self.get_image(camera_pos=[x, y_, 0.3], cam_target_pos=[x, y_ - 0.00001, 0])
            seg_img = np.array(img[4]).reshape(pixelHeight, pixelWidth)
            seg_img = (seg_img == 1).astype(float)
            img_list.append(seg_img)
            rgb_list.append(rgb)

        return m_r_list, m_t_list, img_list, rgb_list, intrinsic

    def reward(self):
        return NotImplementedError

    def step(self, action):
        return NotImplementedError

    def reset(self):
        return NotImplementedError

    def seed(self, seed=None):
        random.seed(seed)
        return 0

    def render(self, mode='human'):
        obs = self.get_image(camera_pos, cam_target_pos)
        return obs
