from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):

    def __init__(self, m_r_list, m_t_list, img_list, rgb_list, intrinsic):
        self.m_r_list = m_r_list
        self.m_t_list = m_t_list
        self.img_list = img_list
        self.intrinsic = intrinsic
        self.rgb_list = rgb_list

    def __len__(self):
        return len(self.m_r_list)

    def __getitem__(self, idx):
        sample = {
            'r_list': self.m_r_list[idx],
            't_list': self.m_t_list[idx],
            "mask_list": self.img_list[idx],
            "rgb_list": self.rgb_list[idx],
            "intrinsic_list": self.intrinsic
        }

        return sample
