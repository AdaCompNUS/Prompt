import json
import multiprocessing as mp
import numpy as np
import pyflex
import torch
from scipy.spatial.transform import Rotation as R

from core.util.args import get_args
from core.util.util import quatFromAxisAngle, quatFromAxisAngle_batch


def _worker(remote, parent_remote, env_fn_wrapper, gpu_id, gui):
    parent_remote.close()
    env = env_fn_wrapper.var(gpu_id=gpu_id, gui=gui)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'reload':
                env.reload(data)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == 'sample_action':
                remote.send(env.sample_action())
            else:
                raise NotImplementedError
        except EOFError:
            break


class PyFlexEnv():
    """
    This is the base class for the PyFleX based simulators.
    In the base class, the action space and the state space are assumed to be (x,y,z,theta). Theta is the gripper rotation angle
    """

    def __init__(self, name, micro_action_steps, gpu_id, gui):

        self.name = name
        self.micro_action_steps = micro_action_steps
        self.gpu_id = gpu_id
        self.particle_radius = 400
        self.dim_shape_state = 14

        self.clip_force = 0.01
        self.lift_force = 0.035
        self.init_height = 0.2

        # set up gripper
        with open(get_args().gripper_conf) as f:
            gripper = json.load(f)

        self.rest_gripper_dis = gripper['rest_gripper_dis'] + 0.016  # allow some buffer
        self.center = None
        self.halfEdge = np.array(gripper['halfEdge'])
        self.halfEdge_cap = np.array(gripper['halfEdge_cap'])
        pyflex.init(gui)

    def reset(self):

        pyflex.set_scene(0, self.positions.numpy(), self.particle_radius)
        r = R.from_quat(pyflex.get_rigidRotations())
        self.base_ori = r.as_euler('xyz', degrees=False)[1]
        self.clip_steps = 0

        center = np.array([5, 0.0, 5])
        quat = np.array([1., 0., 0., 0.])

        pyflex.add_box(self.halfEdge, center, quat)
        pyflex.add_box(self.halfEdge, center, quat)
        pyflex.add_box(self.halfEdge_cap, center, quat)
        pyflex.step()  # otherwise black screen

        return self._get_obs()

    def load_point_cloud(self):
        cloud = self.pc
        positions = torch.zeros((cloud.shape[0], 4))
        for idx, p in enumerate(cloud):
            # vertical axis z correspond to y in flex
            positions[idx] = torch.tensor([p[0], max(p[2], 0), p[1], 10000])
        return positions

    def reload(self, pc_path):

        self.pc = torch.tensor(np.loadtxt(pc_path)).float()
        self.center = self.pc.mean(0)
        self.positions = self.load_point_cloud()

    def step(self, action):

        pyflex.set_friction(0.03)  # enable rotation effect
        for i in range(int(self.micro_action_steps / 3), self.micro_action_steps):
            if i == int(self.micro_action_steps * 2 / 3):
                pyflex.set_friction(1.0)  # enable grasping

            self.shape_states = self.calc_shape_states(i, self.clip_steps, action)
            if not self.collision_check_gripper():
                # no collision and continue clipping
                self.clip_steps += 1

            pyflex.set_shape_states(self.shape_states)
            pyflex.step()

        done = self._done()
        obs = self._get_obs()
        reward, info = self._reward_func(action)

        return obs, reward, done, info

    def collision_check_gripper(self):
        # check two grippers collision, to avoid penetration
        contact_lines = pyflex.get_contacts()
        conatact_lines = torch.tensor(contact_lines).float().view(-1, 6)

        # find out horizontal contact forces
        start_pos = conatact_lines[:, [0, 2]]
        gripper_0 = torch.tensor(self.shape_states[0, [0, 2]]).repeat(conatact_lines.size(0), 1).float()
        gripper_1 = torch.tensor(self.shape_states[1, [0, 2]]).repeat(conatact_lines.size(0), 1).float()

        dis_to_g0 = torch.pow(start_pos - gripper_0, 2).sum(dim=1).sqrt()
        dis_to_g1 = torch.pow(start_pos - gripper_1, 2).sum(dim=1).sqrt()

        g0_touch_num = (dis_to_g0 < self.clip_force).int().sum()
        g1_touch_num = (dis_to_g1 < self.clip_force).int().sum()

        return g0_touch_num > 5 and g1_touch_num > 5

    def calc_shape_states_batch(self, t, clip_steps, action, device="cuda"):
        # unit test pass
        # robot tip graping point: xyh theta. h is height in flex.
        action = action.to(device).float()
        x, y, h, alpha = action[:, 0], action[:, 1], action[:, 2], action[:, 3]

        time_step = self.micro_action_steps
        half_rest_gripper_dis = self.rest_gripper_dis / 2.
        alpha = torch.atan2(torch.sin(torch.deg2rad(alpha)), torch.cos(torch.deg2rad(alpha)))  # normalize angle

        a, b = torch.cos(alpha) * half_rest_gripper_dis, torch.sin(alpha) * half_rest_gripper_dis
        x_init_0, z_init_0 = x + a, y + b
        x_init_1, z_init_1 = x - a, y - b

        progress = min(clip_steps / (time_step / 3), 1)
        e_0_curr = torch.cat(((x_init_0 - progress * a - torch.cos(alpha) * self.clip_force).view(-1, 1),
                              h.view(-1, 1),
                              (z_init_0 - progress * b - torch.sin(alpha) * self.clip_force).view(-1, 1)), dim=1)
        e_1_curr = torch.cat(((x_init_1 + progress * a + torch.cos(alpha) * self.clip_force).view(-1, 1),
                              h.view(-1, 1),
                              (z_init_1 + progress * b + torch.sin(alpha) * self.clip_force).view(-1, 1)), dim=1)
        e_0_last = torch.cat(((x_init_0 - progress * a).view(-1, 1),
                              h.view(-1, 1),
                              (z_init_0 - progress * b).view(-1, 1)), dim=1)
        e_1_last = torch.cat(((x_init_1 + progress * a).view(-1, 1),
                              h.view(-1, 1),
                              (z_init_1 + progress * b).view(-1, 1)), dim=1)
        if t > time_step * 2 / 3:
            t = t - time_step * 2 / 3
            progress = t / (time_step / 3)
            e_0_curr[:, 1] = progress * self.init_height + h + self.lift_force
            e_1_curr[:, 1] = progress * self.init_height + h + self.lift_force

            e_0_last[:, 1] = progress * self.init_height + h
            e_1_last[:, 1] = progress * self.init_height + h

        quat = quatFromAxisAngle_batch(torch.arctan(torch.cos(alpha) / torch.sin(alpha)))

        e_2_curr = torch.cat((((e_0_last[:, 0] + e_1_last[:, 0]) / 2).view(-1, 1),
                              (e_0_last[:, 1] + self.halfEdge[1] + self.halfEdge_cap[1] - 0.01).view(-1, 1),
                              ((e_0_last[:, 2] + e_1_last[:, 2]) / 2).view(-1, 1)), dim=1)
        e_2_last = torch.cat((((e_0_last[:, 0] + e_1_last[:, 0]) / 2).view(-1, 1),
                              (e_0_last[:, 1] + self.halfEdge[1] + self.halfEdge_cap[1]).view(-1, 1),
                              ((e_0_last[:, 2] + e_1_last[:, 2]) / 2).view(-1, 1)), dim=1)

        return e_0_curr, e_0_last, e_1_curr, e_1_last, e_2_curr, e_2_last, quat

    def calc_shape_states(self, t, clip_steps, action):
        x, y, h, alpha = action

        time_step = self.micro_action_steps
        half_rest_gripper_dis = self.rest_gripper_dis / 2.
        alpha = np.arctan2(np.sin(np.radians(alpha)), np.cos(np.radians(alpha)))  # normalize angle

        a, b = np.cos(alpha) * half_rest_gripper_dis, np.sin(alpha) * half_rest_gripper_dis
        x_init_0, z_init_0 = x + a, y + b
        x_init_1, z_init_1 = x - a, y - b

        progress = min(clip_steps / (time_step / 3), 1)
        e_0_curr = np.array([x_init_0 - progress * a - np.cos(alpha) * self.clip_force, h,
                             z_init_0 - progress * b - np.sin(alpha) * self.clip_force])
        e_1_curr = np.array([x_init_1 + progress * a + np.cos(alpha) * self.clip_force, h,
                             z_init_1 + progress * b + np.sin(alpha) * self.clip_force])
        e_0_last = np.array(
            [x_init_0 - progress * a, h,
             z_init_0 - progress * b])
        e_1_last = np.array(
            [x_init_1 + progress * a, h,
             z_init_1 + progress * b])
        if t > time_step * 2 / 3:
            t = t - time_step * 2 / 3
            e_0_curr[1] = t / (time_step / 3) * self.init_height + h + self.lift_force
            e_1_curr[1] = t / (time_step / 3) * self.init_height + h + self.lift_force

            e_0_last[1] = t / (time_step / 3) * self.init_height + h - 0.02
            e_1_last[1] = t / (time_step / 3) * self.init_height + h - 0.02

        states = np.zeros((3, self.dim_shape_state))

        quat = quatFromAxisAngle(np.array([0., 1., 0.]), np.arctan(np.cos(alpha) / np.sin(alpha)))
        states[0, :3] = e_0_curr
        states[0, 3:6] = e_0_last
        states[0, 6:10] = quat
        states[0, 10:14] = quat

        states[1, :3] = e_1_curr
        states[1, 3:6] = e_1_last
        states[1, 6:10] = quat
        states[1, 10:14] = quat

        states[2, :3] = np.array(

            [(e_0_last[0] + e_1_last[0]) / 2, e_0_last[1] + self.halfEdge[1] + self.halfEdge_cap[1] - 0.01,
             (e_0_last[2] + e_1_last[2]) / 2])
        states[2, 3:6] = np.array(
            [(e_0_last[0] + e_1_last[0]) / 2, e_0_last[1] + self.halfEdge[1] + self.halfEdge_cap[1],
             (e_0_last[2] + e_1_last[2]) / 2])

        states[2, 6:10] = quat
        states[2, 10:14] = quat

        return states

    def _reward_func(self, action):
        raise NotImplementedError

    def _move_particle(self, act):
        raise NotImplementedError

    def _done(self):
        raise NotImplementedError

    def _get_obs(self):
        return None


class GraspEnv(PyFlexEnv):
    fail_reward = -1000000

    def __init__(self, name="Nvidia FleX Grasp Simulation", micro_action_steps=60, gpu_id=0, gui=0):
        super().__init__(name, micro_action_steps, gpu_id, gui)

    def _reward_func(self, action):
        new_pos = pyflex.get_positions()
        new_pos = new_pos.reshape(self.positions.shape)
        new_pos = torch.tensor(new_pos).float()[:, :3]
        height_old = self.positions[:, 1].max().item()
        height_new = new_pos[:, 1].max().item()

        if height_new - height_old < 0.12:
            return self.fail_reward, ""

        # rotation, evaluate the robustness
        move_vec = new_pos - self.positions[:, :3]
        move_vec_x, move_vec_z = move_vec[:, 0], move_vec[:, 2]
        error = move_vec_x.square().sum() + move_vec_z.square().sum()

        return error.item() * -1, ""

    def _done(self):
        return True

    def _get_obs(self):
        return None

    def _get_action_space(self):
        return self.action_space

    def _get_observation_space(self):
        return self.observation_space

    def get_spaces(self):
        return self._get_observation_space(), self._get_action_space()

    def sample_action(self):
        return self.action_space.sample()


class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        import cloudpickle
        self.var = cloudpickle.loads(obs)


class VectorEnv():
    """
    This implements the vectorized environment for the pyflex environments. The implementation follows OpenAI
    baselines: https://github.com/openai/baselines
    """

    def __init__(self, env_fns, start_method=None, gpu_id=0, gui=0):
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'

        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn), gpu_id, gui * int(len(self.processes) < 1))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def sample_actions(self):
        for remote in self.remotes:
            remote.send(('sample_action', None))
        self.waiting = True

        results = [remove.recv() for remove in self.remotes]
        self.waiting = False
        return np.stack(results)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reload(self, action):
        # reload new particles from file
        for remote in self.remotes:
            remote.send(('reload', action))

    def reloads(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('reload', action))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True
