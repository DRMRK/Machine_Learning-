import torch

class WineDataset:
    
    def __init__(self,features, target):
        """
        :param features: input features dataframe
        :param target: target dataframe  
        """
        self.x = torch.tensor(features).float()
        self.y = torch.tensor(target)
        self.n_samples= features.shape[0]
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]


