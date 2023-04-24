import torch

class MLP_Basic(torch.nn.Module):
    def __init__(self, feature_size, mri_dimension=19004):
        super(MLP_Basic, self).__init__()
        self.feature_size = feature_size
        self.mri_dimension = mri_dimension

        self.hid1 = torch.nn.Linear(self.feature_size, self.mri_dimension // 3)  
        self.hid2 = torch.nn.Linear(self.mri_dimension // 3, self.mri_dimension // 2)
        self.oupt = torch.nn.Linear(self.mri_dimension // 2, self.mri_dimension)
    
    def forward(self, src):
        src = src.view(-1, src.shape[1] * src.shape[2])
        z = torch.relu(self.hid1(src))
        z = torch.relu(self.hid2(z))
        z = self.oupt(z)
        return z