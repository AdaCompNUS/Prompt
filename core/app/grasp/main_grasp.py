import os
import sys
import time

import gym
import numpy as np
import torch

sys.path.append(os.getcwd())

from core.reconstruction.fc_gen import FCGenerator
from core.reconstruction.recon_service import generate_point_cloud
from core.tracking.tracking import track
from core.util.args import get_args
from core.util.util import create_new_step_folder, create_new_working_folder


def query_server_action(step_folder):
    # wait for all planning services to finish
    for i in range(num_GPU):
        while not os.path.exists(f"{step_folder}/action_{i}.txt"):
            time.sleep(0.1)

    results = []
    for i in range(num_GPU):
        actions = np.loadtxt(f"{step_folder}/action_{i}.txt").tolist()
        results.append(actions)

    results = sorted(results, key=lambda x: x[4])
    print("results", results)
    return results


args = get_args()
num_GPU = args.num_gpu
env = gym.make('robot:Grasping-v0', GUI=args.visualize_pybullet == 1).unwrapped
for episode in range(1000):
    working_folder_id, working_folder = create_new_working_folder()
    obs = env.reset()

    # estimate object rough pos
    dummy_gen = FCGenerator().cuda()
    pc = dummy_gen().squeeze()
    torch.cuda.empty_cache()
    _, xy_est = track(pc, env, iters=1, init_x=0, init_y=0, var=0.5, num_particles=2000, visualize=False)
    x0, y0, _ = xy_est.cpu().numpy()
    torch.cuda.empty_cache()

    # reconstruction
    step_id, step_folder = create_new_step_folder()
    m_r_list, m_t_list, img_list, rgb_list, intrinsic = env.get_video(step=2)

    start_time = time.time()
    pc = generate_point_cloud(m_r_list, m_t_list, img_list, rgb_list, intrinsic, x0, y0)
    recon_time = time.time() - start_time

    # swap x, y due to pybullet and flex difference
    pc_flex = pc.index_select(1, torch.tensor([1, 0, 2]).cuda())

    # save result for planning services
    start_time = time.time()
    pc_flex_np = pc_flex.clone().cpu().detach().numpy()
    np.savetxt(f"{step_folder}/pc.txt", pc_flex_np)
    np.savetxt(f"{working_folder}/pc.txt", pc_flex_np)

    # visualize in pybullet
    pc_np = pc.clone().cpu().detach().numpy()
    pc_np[:, 0] += 0.2
    env.show_point_cloud(pc_np)
    torch.cuda.empty_cache()

    results = query_server_action(step_folder)
    plan_time = time.time() - start_time
    grasp_action = results[-1][:4]

    # swap x, y back in bullet domain
    grasp_action[0], grasp_action[1] = grasp_action[1], grasp_action[0]
    grasp_action[3] = np.radians(180 - grasp_action[3])  # angle difference between pybullet and flex

    env.visualize_grasp(args, grasp_action)
    _, reward, _, _ = env.step(grasp_action, img_path=f"{working_folder}/action.png")
    print("recon time:", recon_time, "plan time:", plan_time)
    print("Grasp Success" if reward >= 1 else "Grasp Fail")
