import cv2
import torch

from core.util.args import get_args

pixelWidth = 256
pixelHeight = 192
small_num = 1e-8  # to avoid nan


def calc_mask_loss_batch(pos, mask_img):
    args = get_args()
    # high res, pos: N x 2
    mask_x = (pos[:, :, 0] < pixelWidth) * (0 <= pos[:, :, 0])
    mask_y = (pos[:, :, 1] < pixelHeight) * (0 <= pos[:, :, 1])
    mask_in_pic = (mask_x * mask_y).long()  # only count points inside frame.

    gt_pos = None
    for i in range(mask_img.size(0)):
        gt_depth_high = (mask_img[i] > 0.1).int()

        gt_pos_single = torch.nonzero(gt_depth_high)
        gt_pos_single = torch.index_select(gt_pos_single, 1, torch.tensor([1, 0]).cuda())

        sample_idx_list = torch.randint(0, gt_pos_single.size(0), (pos.size(1),))
        gt_pos_single = gt_pos_single[sample_idx_list].unsqueeze(0)
        gt_pos = torch.cat((gt_pos, gt_pos_single), dim=0) if gt_pos is not None else gt_pos_single

    p2gt_pos = pos.clone()
    p2gt_pos = p2gt_pos.unsqueeze(2).repeat((1, 1, pos.size(1), 1))
    p2gt_gtpos = gt_pos.clone()
    p2gt_gtpos = p2gt_gtpos.unsqueeze(1).repeat((1, pos.size(1), 1, 1))
    p2gt_dis = (torch.pow(p2gt_gtpos - p2gt_pos, 2).sum(dim=3) + small_num).sqrt()
    p2gt_dis = p2gt_dis.min(2)[0]
    p2gt_loss = (p2gt_dis * mask_in_pic).sum() / pos.size(1)

    gt2p_pos = pos.clone()
    gt2p_pos = gt2p_pos.unsqueeze(1).repeat((1, pos.size(1), 1, 1))
    gt2p_gtpos = gt_pos.clone()
    gt2p_gtpos = gt2p_gtpos.unsqueeze(2).repeat((1, 1, pos.size(1), 1))
    gt2p_dis = (torch.pow(gt2p_gtpos - gt2p_pos, 2).sum(dim=3) + small_num).sqrt()
    min_k = 1.0 / ((1.0 / gt2p_dis).topk(args.k)[0].mean(2))
    gt2p_loss = (min_k * mask_in_pic).sum() / pos.size(1)

    debug = False
    if debug:
        # visualization
        pos = pos[0][mask_in_pic[0] == 1]
        pos = pos.view(-1, 2).long()
        gt_depth_high = (mask_img[0] > 0.1).int()
        gt_depth_high = 1 - gt_depth_high
        w_img_high = torch.ones_like(mask_img[0]).cuda()

        gt_depth_high_sampled = w_img_high.clone()
        gt_depth_high_sampled[gt_pos[0, :, 1].view(-1), gt_pos[0, :, 0].view(-1)] = 0
        pred_img_high = w_img_high.clone()
        pred_img_high[pos[:, 1].view(-1), pos[:, 0].view(-1)] = 0

        pred_img_high_val = pred_img_high.cpu().numpy()
        gt_depth_high_val = gt_depth_high.float().cpu().numpy()
        gt_depth_high_sampled_val = gt_depth_high_sampled.clone().float().cpu().detach().numpy()
        cv2.imshow("pred_img_high_val", pred_img_high_val)
        cv2.imshow("gt_depth_high_val", gt_depth_high_val)
        cv2.imshow("gt_depth_high_sampled_val", gt_depth_high_sampled_val)
        cv2.waitKey(0)

    return p2gt_loss + gt2p_loss


def calc_projection_batch(dotPos, intrinsic, m_r, m_t):
    batch_size = m_t.size(0)

    # calc projection
    calc_projection_const = torch.ones((1, dotPos.size(0))).cuda()
    dotPos = dotPos.permute((1, 0))
    dotPos = torch.cat((dotPos, calc_projection_const.clone()), dim=0)

    m2 = torch.matmul(m_r, dotPos)
    m2 = torch.matmul(m_t, m2)
    new_p = torch.matmul(intrinsic, m2) + small_num
    ratio = torch.ones((batch_size, 1, dotPos.size(1))).cuda() / new_p[:, 2, :].view(batch_size, 1, new_p.size(2))
    new_p = new_p * ratio
    new_p = new_p[:, :2, :]
    new_p = new_p.permute((0, 2, 1))
    return new_p


def mask_loss_batch(particles, mask_imgs, intrinsics, rotate_matrixs, trans_vecs):
    pos = calc_projection_batch(particles, intrinsics, rotate_matrixs, trans_vecs)
    loss = calc_mask_loss_batch(pos=pos, mask_img=mask_imgs)

    return loss / mask_imgs.size(0)
