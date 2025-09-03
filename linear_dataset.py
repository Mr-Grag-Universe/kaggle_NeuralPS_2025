import torch
import warnings
from torch.utils.data import Dataset
from tqdm import tqdm
from .process_data import *

def _get_scalar_features(scalar_features_df, planet_ids):
    res = []
    try:
        for pid in planet_ids:
            scalar_features = scalar_features_df[scalar_features_df['planet_id'] == pid].values[0,1:]
            res.append(torch.tensor(scalar_features, dtype=torch.float32))
    except:
        pid = planet_ids
        scalar_features = scalar_features_df[scalar_features_df['planet_id'] == pid].values[0,1:]
        return torch.tensor(scalar_features, dtype=torch.float32)
    return torch.stack(res)

class LinearDataset(Dataset):
    def __init__(self, dataset, denoise_model, scalar_features_df, train=True, device=None):
        self.dataset = dataset
        self.denoise_model = denoise_model.eval()
        self.scalar_features_df = scalar_features_df
        self.train = train
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]

        full_spectrum = torch.tensor(item['signal'], device=self.device, dtype=torch.float).permute(1, 0, 2)
        noisy_target = torch.tensor(item['noisy_target'], device=self.device, dtype=torch.float)
        with torch.no_grad():
            clear_spectrum = self.denoise_model(full_spectrum.unsqueeze(0), det_array=noisy_target.unsqueeze(0))
        clear_target = get_target(clear_spectrum.permute(0, 2, 1, 3)).to(self.device).float()
        scalar_features = _get_scalar_features(self.scalar_features_df, item['planet_id']).to(self.device).float()
        
        inputs = torch.cat((clear_target[0], 
                            noisy_target, 
                            full_spectrum.sum(axis=(0, 2)), # flux
                            full_spectrum.sum(0).mean(0), # stellar spectrum
                            scalar_features))
        if self.train:
            second = item['target']
        else:
            second = item['planet_id']
        return inputs, second

def get_gp_train_xy(dataset):
    X, Y = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, y in tqdm(dataset):
            X.append(x)
            Y.append(y)
    return torch.stack(X), torch.stack(Y)

def get_gp_test_x(dataset):
    X = []
    planet_ids = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, planet_id in tqdm(dataset):
            X.append(x)
            planet_ids.append(planet_id)
    return torch.stack(X), planet_ids