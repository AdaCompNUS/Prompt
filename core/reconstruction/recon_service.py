import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.getcwd())
from core.reconstruction.cnn_gen import CNNGenerator
from core.reconstruction.fc_gen import FCGenerator
from core.util.args import get_args
from core.reconstruction.recon_dataset import ReconstructionDataset
from core.reconstruction.recon_loss import mask_loss_batch
from core.util.util import get_lr
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.autograd.set_detect_anomaly(True)


def optimize_cloud(cloud):
    # remove outliner
    gt2p_pos = cloud.clone()
    gt2p_pos = gt2p_pos.unsqueeze(0).repeat((cloud.size(0), 1, 1))
    gt2p_gtpos = cloud.clone()
    gt2p_gtpos = gt2p_gtpos.unsqueeze(1).repeat((1, cloud.size(0), 1))
    gt2p_dis = torch.pow(gt2p_gtpos - gt2p_pos, 2).sum(dim=2).sqrt()
    index_x = torch.arange(cloud.size(0)).cuda()
    gt2p_dis[index_x, index_x] = 100
    min_k = 1.0 / ((1.0 / gt2p_dis).topk(15)[0].mean(1))

    avg = min_k.mean()
    pos = cloud[min_k < avg * 1.3]

    return pos


def underground_loss(pos):
    loss = pos[:, 2].clamp_max(0).abs().sum()
    return loss


def init_model(args):
    # generator
    if args.gen_type == "CNN":
        gen_net = CNNGenerator(num_p=args.num_p)  # rgb channels
        optimizer = torch.optim.Adam(gen_net.parameters(), lr=args.learning_rate_CNN)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience,
                                      threshold=args.lr_threshold, verbose=True, min_lr=args.min_lr)
    else:
        gen_net = FCGenerator(init_var=args.init_var_FC, num_p=args.num_p)
        optimizer = torch.optim.Adam(gen_net.parameters(), lr=args.learning_rate_FC)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience,
                                      threshold=args.lr_threshold, verbose=True, min_lr=args.min_lr)

    gen_net = gen_net.cuda().float()

    return gen_net, optimizer, scheduler


def generate_point_cloud(m_r_list, m_t_list, img_list, rgb_list, intrinsic, x, y):
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_set = ReconstructionDataset(m_r_list, m_t_list, img_list, rgb_list, intrinsic)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False)

    gen_net, optimizer, scheduler = init_model(args)

    print("Reconstruction service is running...")
    start_time = time.time()
    for epoch in range(args.num_epoch):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            rgb_list = data['rgb_list'].cuda().float()
            particles = gen_net(rgb_list).squeeze()
            particles[:, 0] += x
            particles[:, 1] += y
            particles[:, 2] += 0.05  # no points under ground

            # calc loss
            loss_mask = mask_loss_batch(particles,
                                        data['mask_list'].cuda().float(),
                                        data['intrinsic_list'].cuda().float(),
                                        data['r_list'].cuda().float(),
                                        data["t_list"].cuda().float())

            under_loss = underground_loss(particles)
            loss = loss_mask + under_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 0.005)
            optimizer.step()
            scheduler.step(loss)

        print('[%5d / %5d] loss: %.3f lr: %.10f'
              % (epoch, args.num_epoch, loss_mask.clone().item(),
                 get_lr(optimizer)))

    print(f"Reconstruction finishes in {time.time() - start_time:.2f} seconds")
    # save results
    particles = optimize_cloud(particles)
    particles = optimize_cloud(particles)

    return particles
