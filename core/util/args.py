import argparse
import sys


def get_args():
    # Hyperparameters
    parser = argparse.ArgumentParser(description='Reconstruction')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--task', type=str, default='grasp', help='task to perform')
    parser.add_argument('--gen_type', type=str, default='FC', help='generator type')
    parser.add_argument('--num_view', type=int, default=200, metavar='E', help='number of views of object')
    parser.add_argument('--obj_per_cat', type=int, default=50, metavar='E', help='no objects per category')
    parser.add_argument('--k', type=int, default=100, metavar='E', help='no neighbors in Chamfer Distance')
    parser.add_argument('--num_p', type=int, default=1500, metavar='E', help='no particles used')

    parser.add_argument('--num_epoch', type=int, default=12, metavar='E', help='Total number of episodes')
    parser.add_argument('--batch_size', type=int, default=10, metavar='B', help='Batch size')

    parser.add_argument('--learning_rate_FC', type=float, default=2e-3, metavar='α', help='Learning rate')
    parser.add_argument('--learning_rate_CNN', type=float, default=4e-4, metavar='α', help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=2e-7, metavar='α', help='Learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.2, metavar='α', help='lr decay rate')
    parser.add_argument('--lr_patience', type=float, default=15, metavar='α', help='lr decay rate')
    parser.add_argument('--lr_threshold', type=float, default=2e-1, metavar='α', help='lr decay rate')
    parser.add_argument('--init_var_FC', type=float, default=0.15, metavar='α', help='Learning rate')

    parser.add_argument('--gripper_conf', type=str, default='core/conf/franka.json', help='gripper config')

    # planning
    parser.add_argument('--gpu_id', type=int, default=0, metavar='B', help='GPU id, used when start planning service')
    parser.add_argument('--num_gpu', type=int, default=1, metavar='B', help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=20, metavar='B', help='Number of workers for each gpu')
    parser.add_argument('--visualize_flex', type=int, default=0, metavar='B', help='visualize flex')
    parser.add_argument('--visualize_pybullet', type=int, default=1, metavar='B', help='visualize pybullet')

    args = parser.parse_args(sys.argv[1:])

    return args
