import torch

from core.tracking.simple_mpc import run_simple_mpc


def calc_mr_mt_batch(para):
    para.cuda()
    b_size = para.size(0)
    t = torch.zeros((b_size, 3)).cuda()
    t[:, :2] = para[:, :2]
    gamma_ = para[:, 2]

    r_matrix = torch.eye(3).unsqueeze(0).repeat((b_size, 1, 1)).cuda()
    r_matrix[:, 0, 0] = torch.cos(gamma_)
    r_matrix[:, 0, 1] = torch.sin(gamma_)
    r_matrix[:, 1, 0] = -torch.sin(gamma_)
    r_matrix[:, 1, 1] = torch.cos(gamma_)

    m_r = torch.zeros((b_size, 4, 4)).float().cuda()
    m_r[:, :3, :3] = r_matrix
    m_r[:, 3, 3] = 1

    m_t = torch.zeros((b_size, 3, 4)).cuda()
    m_t[:, :3, :3] = torch.eye(3).unsqueeze(0).repeat((b_size, 1, 1))
    m_t[:, :3, 3] = t

    return m_r.float(), m_t.float()


def calc_transformation(dotPos, m_r, m_t):
    # calc projection
    dotPos = dotPos.cuda()
    calc_projection_const = torch.ones((1, dotPos.size(0))).cuda()
    particles = dotPos.permute((1, 0))
    particles = torch.cat((particles, calc_projection_const.clone()), dim=0)

    m2 = torch.matmul(m_r, particles)
    new_p = torch.matmul(m_t, m2)

    if len(new_p.shape) == 2:
        new_p = new_p.permute(1, 0)
    elif len(new_p.shape) == 3:
        new_p = new_p.permute((0, 2, 1))

    return new_p


def track(pc, env, init_x, init_y, iters=1, var=0.07, num_particles=5000, visualize=False, clear_from=0):
    pc = pc.detach()
    center = pc.mean(dim=0).cpu().numpy()
    x0, y0, _ = center
    pc[:, 0] -= x0
    pc[:, 1] -= y0

    ob_data = env.get_video_top_only()
    best_para = run_simple_mpc(num_particles, pc, ob_data, iters=iters, initial_x=[init_x, init_y, 0], var=var)

    m_r, m_t = calc_mr_mt_batch(best_para.view(1, 3))
    pc_est = calc_transformation(pc, m_r.squeeze(), m_t.squeeze())

    if visualize:
        env.show_point_cloud(pc_est.cpu().numpy(), rgbaColor=[0, 1, 1, 1], clear_from=clear_from)

    return pc_est, best_para
