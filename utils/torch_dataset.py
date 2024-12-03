
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
from tqdm import tqdm


class CreateDataset_eeg_fmri(Dataset):
    """
    Important time is last axis. 
    x_tensor - wavelet features - [ch, freq, time] # 
    y_tensor - hand pose - [21, time]
    
    Return: 
        x_crop - [ch, freq, window_size]
        y_crop - [21, window_size]
    """

    def __init__(self, dataset, 
                 window_size=1024, 
                 random_sample=False, 
                 sample_per_epoch=None, 
                 to_many = False):

        self.x, self.y = dataset
        self.WINDOW_SIZE = window_size
        self.start_max = self.x.shape[-1] - window_size - 1 

        self.random_sample = random_sample
        self.sample_per_epoch = sample_per_epoch
        self.to_many = to_many
        
    def __len__(self):
        if self.random_sample: 
            return self.sample_per_epoch
        return self.start_max

    def __getitem__(self, idx):
        
        if self.random_sample: 
            idx  = np.random.randint(0, self.start_max)
        
        start , end = idx, idx+self.WINDOW_SIZE
        eeg = self.x[..., start:end]
        
        if self.to_many:
            fmri = self.y[..., start:end]
        else:
            fmri = self.y[..., end-1]
        return (eeg, fmri)
    
    def get_full_dataset(self,inp_stride=1, step=1):
        """
        step - step between starting points neighbour points 
        inp_stride - take input with some stride( downsample additional). 
        """
        x_list = []
        y_list = []
        for idx in tqdm(range(0, len(self), step)):
            data = self[idx]
            x_list.append(data[0][..., ::inp_stride])
            y_list.append(data[1])
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        return x_list, y_list 
        
class CreateDeepONetDataset(Dataset):
    def __init__(self, dataset, 
                 window_size=1024, 
                 random_sample=False, 
                 sample_per_epoch=None, 
                 to_many=False):

        self.x, self.y = dataset
        self.WINDOW_SIZE = window_size
        self.start_max = self.x.shape[-1] - window_size - 1 

        self.random_sample = random_sample
        self.sample_per_epoch = sample_per_epoch
        self.to_many = to_many
        
    def __len__(self):
        if self.random_sample: 
            return self.sample_per_epoch
        return self.start_max

    def __getitem__(self, idx):
        if self.random_sample: 
            idx = np.random.randint(0, self.start_max)
        
        start, end = idx, idx+self.WINDOW_SIZE
        
        # EEG data
        eeg = self.x[..., start:end]
        
        # fMRI data
        if self.to_many:
            fmri = self.y[..., start:end]
        else:
            fmri = self.y[..., end-1]
        

        # Create trunk input - here I'm using normalized time as a simple example
        # This can be modified based on your specific needs
        trunk_input = torch.FloatTensor(self.WINDOW_SIZE * 8).uniform_(-2, 2)
        #trunk_input = torch.linspace(start / self.x.shape[-1], 
                                    #end / self.x.shape[-1], 
                                   # self.WINDOW_SIZE * 8).float()

        #trunk_input = trunk_input.repeat(8)
        #trunk_input = trunk_input.unsqueeze(0)  # Shape: (1, 16384) -> adding batch dimension

        #trunk_input = torch.tensor([(idx + self.WINDOW_SIZE/2) / self.x.shape[-1]], dtype=torch.float32)
        #print(f"EEG SHAPE{eeg.shape}")
        #print(f"TRUNK IN DATASET SHAPE{trunk_input.shape}")

        return (eeg, fmri, trunk_input)
    
    def get_full_dataset(self, inp_stride=1, step=1):
        x_list = []
        y_list = []
        trunk_list = []
        
        for idx in tqdm(range(0, len(self), step)):
            data = self[idx]
            x_list.append(data[0][..., ::inp_stride])
            y_list.append(data[1])
            trunk_list.append(data[2])
        
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        trunk_list = np.array(trunk_list)
        
        return x_list, y_list, trunk_list