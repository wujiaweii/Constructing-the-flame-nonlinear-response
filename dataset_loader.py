import pandas as pd
import torch
from torch.utils import data


def get_loader(inputdata,label, batch_size, num_workers):
    """Builds and returns Dataloader."""

    inputdata=torch.tensor(inputdata,dtype=torch.float32)
    label=torch.tensor(label,dtype=torch.float32)

    dataset= data.TensorDataset(inputdata,label)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader