import os

import numpy as np
import torch

from core.util.args import get_args


def get_working_folder():
    task = get_args().task
    folder = f"core/app/{task}/data_folder"
    if not os.path.exists(folder):
        os.mkdir(folder)

    work_folders = [int(f.split("_")[1]) for f in os.listdir(folder) if
                    os.path.isdir(os.path.join(folder, f)) and "data_" in f]

    working_folder_id = max(work_folders) if len(work_folders) > 0 else -1
    working_folder = f"core/app/{task}/data_folder/data_{working_folder_id}"

    return working_folder_id, working_folder


def get_step_folder():
    working_folder_id, working_folder = get_working_folder()
    if working_folder_id < 0:
        return -1, f"{working_folder}/step_-1"

    work_folders = [int(f.split("_")[1]) for f in os.listdir(working_folder) if
                    os.path.isdir(os.path.join(working_folder, f)) and "step_" in f]

    step_id = max(work_folders) if len(work_folders) > 0 else -1
    step_folder = f"{working_folder}/step_{step_id}"

    return step_id, step_folder


def create_new_working_folder():
    task = get_args().task
    working_folder_id, _ = get_working_folder()
    working_folder_id += 1
    working_folder = f"core/app/{task}/data_folder/data_{working_folder_id}"
    os.mkdir(working_folder)

    return working_folder_id, working_folder


def create_new_step_folder():
    working_folder_id, working_folder = get_working_folder()
    step_id, step_folder = get_step_folder()
    step_id += 1
    step_folder = f"{working_folder}/step_{step_id}"
    os.mkdir(step_folder)

    return step_id, step_folder


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def quatFromAxisAngle_batch(angle, device="cuda:0"):
    axis = torch.tensor([0., 1., 0.]).float().to(device)
    axis = axis.unsqueeze(0).repeat((angle.size(0), 1))
    angle = angle.to(device)

    half = angle * 0.5
    w = torch.cos(half).view(-1, 1)

    sin_theta_over_two = torch.sin(half)
    axis *= sin_theta_over_two.view(-1, 1)
    quat = torch.cat((axis, w), dim=1)

    return quat
