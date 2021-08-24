import os
import sys
import time
import numpy as np

sys.path.append(os.getcwd())

from core.util.args import get_args
from core.planning.grasp.mpc_planner import MPCPlanner
from core.planning.grasp.grasp_env import GraspEnv, VectorEnv
from core.util.util import get_working_folder, get_step_folder

top_candidates = 1
optimisation_iters = 6


def transition_model(envs, actions):
    envs.reset()
    obs, reward, done, info = envs.step(actions)
    print(info)

    return reward


def plan(particle_envs, pc_path):
    mpc = MPCPlanner(action_size=3, planning_horizon=1, optimisation_iters=optimisation_iters,
                     candidates=num_workers, top_candidates=top_candidates, transition_model=transition_model)
    action_mean, action_std_dev, value, returns = mpc.forward(particle_envs, pc_path)
    return action_mean, action_std_dev, value, returns


if __name__ == "__main__":

    args = get_args()
    task = args.task
    gpu_id = args.gpu_id
    num_workers = args.num_workers

    particle_envs = [GraspEnv for _ in range(num_workers)]
    particle_envs = VectorEnv(particle_envs, gpu_id=gpu_id, gui=args.visualize_flex)

    while True:
        step_id, step_folder = get_step_folder()
        working_folder_id, working_folder = get_working_folder()
        pc_path = f"{step_folder}/pc.txt"
        action_path = f"{step_folder}/action_{gpu_id}.txt"
        if not os.path.exists(pc_path):
            # reconstruction is not ready yet
            time.sleep(0.5)
            continue

        if os.path.exists(action_path):
            # planning finished
            time.sleep(0.5)
            continue

        start_time = time.time()
        particle_envs.reload(pc_path)
        particle_envs.reset()

        action_mean, action_std_dev, value, returns = plan(particle_envs, pc_path)
        action_mean = action_mean.cpu().numpy()

        print("planned action:", action_mean, value)
        np.savetxt(action_path, np.append(action_mean, value.cpu().item()))
