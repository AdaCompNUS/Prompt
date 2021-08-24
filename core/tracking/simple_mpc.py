import time

import cv2
import numpy as np
import torch
from numpy.random import randn


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


def calc_projection_batch(pc, intrinsic, m_r, m_t):
    batch_size = pc.size(0)
    num_p = pc.size(1)

    # calc projection
    calc_projection_const = torch.ones((batch_size, 1, num_p)).cuda()
    pc = pc.permute((0, 2, 1))
    pc = torch.cat((pc, calc_projection_const), dim=1)

    m2 = torch.matmul(m_r, pc)
    m2 = torch.matmul(m_t, m2)
    new_p = torch.matmul(intrinsic, m2)
    ratio = torch.ones((batch_size, 1, num_p)).cuda() / new_p[:, 2, :].view(batch_size, 1, num_p)
    new_p = new_p * ratio
    new_p = new_p[:, :2, :]
    new_p = new_p.permute((0, 2, 1))
    return new_p


def calc_proj_error(pc, img):
    # img: y _ x
    b_size = pc.size(0)
    num_p = pc.size(1)
    pc = pc.long()
    proj_imgs = torch.zeros((b_size, img.size(0), img.size(1))).cuda()

    pc_num_p_b4 = pc.abs().sum(1).sum(1)
    pc[:, :, 0] = torch.clamp(pc[:, :, 0], 0, img.size(1) - 1)
    pc[:, :, 1] = torch.clamp(pc[:, :, 1], 0, img.size(0) - 1)
    pc_clamp_error = pc_num_p_b4 - pc.abs().sum(1).sum(1)

    idx_tmp = torch.range(0, b_size - 1).cuda().long().view(b_size, 1).repeat(1, num_p)
    proj_imgs[idx_tmp, pc[:, :, 1], pc[:, :, 0]] = 1

    debug = False
    if debug:
        proj_imgs_np = proj_imgs.cpu().numpy()
        for i in range(proj_imgs_np.shape[0]):
            img_proj = proj_imgs_np[i]
            cv2.imshow("proj", img_proj)
            cv2.waitKey(0)

    proj_error = (proj_imgs - img).abs().view(b_size, -1).sum(1)
    proj_error += pc_clamp_error

    debug = False
    if debug:
        best_idx = torch.argmin(proj_error)
        img_proj = proj_imgs[best_idx].cpu().numpy()
        img_np = img.cpu().numpy()
        cv2.imshow("proj", img_proj)
        cv2.imshow("img_np", img_np)
        cv2.waitKey(0)

    return proj_error


def roll_out(paras, ob_data, pc):
    from core.tracking.tracking import calc_mr_mt_batch, calc_transformation

    m_r_list, m_t_list, img_list, rgb_list, intrinsic = ob_data
    img_list = torch.tensor(img_list).cuda().float()
    intrinsic = torch.tensor(intrinsic).cuda().float()
    m_r_list = torch.tensor(m_r_list).cuda().float()
    m_t_list = torch.tensor(m_t_list).cuda().float()

    paras = torch.tensor(paras).cuda().float()

    m_r, m_t = calc_mr_mt_batch(paras)
    pc_moved = calc_transformation(pc, m_r, m_t)
    proj_error_all = torch.zeros((paras.size(0),)).cuda()
    for i in range(len(m_r_list)):
        pc_proj = calc_projection_batch(pc_moved, intrinsic, m_r_list[i], m_t_list[i])
        proj_error = calc_proj_error(pc_proj, img_list[i])
        proj_error_all += proj_error

    best_para = paras[torch.argmin(proj_error_all)]

    return best_para


def run_simple_mpc(N, pc, ob_data, iters=1, initial_x=None, var=0.07):
    # create particles and weights

    particles = create_gaussian_particles(mean=initial_x, std=(var, var, np.pi), N=N)

    for x in range(iters):
        # incorporate measurements
        start_time = time.time() * 1000
        best_para = roll_out(particles, ob_data=ob_data, pc=pc)
        print("best_para update in ", time.time() * 1000 - start_time)

        particles = create_gaussian_particles(mean=best_para.cpu().numpy(),
                                              std=(var / (x + 1), var / (x + 1), np.pi / (x + 1)), N=N)

    return best_para
