import json
import numpy as np
import torch

from core.planning.grasp.grasp_env import PyFlexEnv, GraspEnv
from core.util.args import get_args


class MPCPlanner():
    __constants__ = ['action_size', 'planning_horizon',
                     'optimisation_iters', 'candidates', 'top_candidates']

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model):
        self.transition_model = transition_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

        with open(get_args().gripper_conf) as f:
            self.gripper = json.load(f)

    def get_grasp_height(self, env, pc, x, y):
        pc = pc.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = x.view(-1, 1)
        y = y.view(-1, 1)

        mask_x = (pc[:, :, 0] < x + 0.01) * (x - 0.01 <= pc[:, :, 0])
        mask_y = (pc[:, :, 1] < y + 0.01) * (y - 0.01 <= pc[:, :, 1])

        max_h = (pc * (mask_x * mask_y).int().unsqueeze(2)).max(dim=1)[0][:, 2]
        max_h = (max_h - self.gripper['finger_depth'] + 0.005).clamp(min=0.005)

        return max_h + env.halfEdge[1]

    def collision_check(self, env, points, action):

        _, g0, _, g1, _, h0, quat = env.calc_shape_states_batch(int(env.micro_action_steps / 3), 0, action)
        halfEdge = torch.tensor(env.halfEdge).float().cuda()
        halfEdge_hor = torch.tensor(env.halfEdge_cap).float().cuda()
        points = points.unsqueeze(0).repeat(action.size(0), 1, 1)

        # convert (x, h, y) into (x,y,h)
        g0 = g0[:, [0, 2, 1]]
        g1 = g1[:, [0, 2, 1]]
        h0 = h0[:, [0, 2, 1]]
        halfEdge = halfEdge[[0, 2, 1]]
        halfEdge_hor = halfEdge_hor[[0, 2, 1]]

        g0_low = (g0 - halfEdge).unsqueeze(1)
        g0_high = (g0 + halfEdge).unsqueeze(1)

        g1_low = (g1 - halfEdge).unsqueeze(1)
        g1_high = (g1 + halfEdge).unsqueeze(1)

        h0_low = (h0 - halfEdge_hor).unsqueeze(1)
        h0_high = (h0 + halfEdge_hor).unsqueeze(1)

        g0_low_filtered = ((points - g0_low) > 0).int().prod(dim=2)
        g0_high_filtered = ((g0_high - points) > 0).int().prod(dim=2)
        g0_filtered = g0_low_filtered * g0_high_filtered

        g1_low_filtered = ((points - g1_low) > 0).int().prod(dim=2)
        g1_high_filtered = ((g1_high - points) > 0).int().prod(dim=2)
        g1_filtered = g1_low_filtered * g1_high_filtered

        h0_low_filtered = ((points - h0_low) > 0).int().prod(dim=2)
        h0_high_filtered = ((h0_high - points) > 0).int().prod(dim=2)
        h0_filtered = h0_low_filtered * h0_high_filtered

        collisions = g0_filtered.sum(1) + g1_filtered.sum(1) + h0_filtered.sum(1)
        action_no_collision = action[collisions <= 2]

        return action_no_collision

    def set_actions_fix_size(self, action):
        if action.size(0) < self.candidates:
            action_ = action[0].view(1, 4).repeat(self.candidates - action.size(0), 1)
            action = torch.cat((action, action_), dim=0)

        action = action[:self.candidates]
        return action

    def uniform_sample_from_prior(self, env, pc):

        sample_size = 5000
        sample_idx = torch.randint(0, pc.size(0), (sample_size,)).cuda()
        action_xy = pc[sample_idx][:, :2]
        grasp_height = self.get_grasp_height(env, pc, action_xy[:, 0], action_xy[:, 1]).view(-1, 1)

        action_theta = 90 * torch.randn(sample_size, 1).cuda()
        action = torch.cat((action_xy, grasp_height, action_theta), dim=1)

        action = self.collision_check(env, pc, action)
        action = self.set_actions_fix_size(action)
        return action

    def forward(self, envs, pc_path):
        action_mean = torch.tensor([0, 0, 0]).cuda()
        action_std_dev = torch.tensor([0.03, 0.03, 90]).cuda()

        pc = torch.tensor(np.loadtxt(pc_path)).float().cuda()
        env = PyFlexEnv(name="", micro_action_steps=60, gpu_id=0, gui=0)
        sample_size = 500

        start_point_found = False
        for i in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            # actions = action_mean + action_std_dev * torch.randn(self.candidates, self.action_size)
            action_std_dev_curr = action_std_dev / (i + 2)
            if start_point_found:

                # action_proposed = action_mean + action_std_dev * torch.randn(sample_size, self.action_size)
                actions = action_mean + action_std_dev_curr * torch.randn(sample_size, self.action_size).cuda()
                grasp_height = self.get_grasp_height(env, pc, actions[:, 0], actions[:, 1]).view(-1, 1)
                actions = torch.cat((actions, grasp_height), dim=1)
                actions = actions[:, [0, 1, 3, 2]]  # convert x,y,theta,h to xyh theta
                actions = self.collision_check(env, pc, actions)
                actions = self.set_actions_fix_size(actions)

            else:
                # uniform samples from prior, which is the xy projection of the point cloud
                actions = self.uniform_sample_from_prior(env, pc)

            # Sample next states
            returns = self.transition_model(envs, actions.cpu().numpy())
            returns = torch.tensor(returns).float().cuda()

            print("candidates:", actions)
            print("action_mean:", action_mean)
            print("action_std_dev:", action_std_dev_curr)
            print("returns:", returns)

            # Re-fit belief to the K best action sequences
            _, topk = returns.topk(self.top_candidates, largest=True, sorted=False)
            best_actions = actions[topk.view(-1)]
            # Update belief with new means and standard deviations

            if returns.sum() > GraspEnv.fail_reward * self.candidates:
                # update when valid results appear
                action_mean = best_actions.mean(dim=0)
                action_mean = action_mean[[0, 1, 3]]  # convert xyh theta to xy theta, as h is generated from xy
                # action_std_dev = best_actions.std(unbiased=False, dim=0)
                start_point_found = True

            # if all candidates success, stop early
            _, top_90_percent = returns.topk(int(0.9 * self.candidates), largest=True, sorted=False)
            if returns[top_90_percent[-1]] > -1.5:
                print("all candidates success, stop early", ' it ', i)
                break

        value = returns[topk.view(-1)].mean()
        best_actions = best_actions.mean(dim=0)
        best_actions[2] -= env.halfEdge[1]  # shift from center of gripper to lower tip
        return best_actions, action_std_dev, value, returns
