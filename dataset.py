from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from .process_data import prepare_signal, get_target, get_noise

class PlanetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.adc_info = pd.read_csv(root_dir.parent/'adc_info.csv')
        self.train_labels = pd.read_csv(root_dir.parent/'train.csv')
        self.axis_info = pd.read_parquet(root_dir.parent / 'axis_info.parquet')
        
        # Получаем список планет (папок с индексами)
        self.planet_folders = [
            root_dir/planet_folder 
            for planet_folder in os.listdir(root_dir) 
            if os.path.isdir(root_dir/planet_folder)
        ]
        
        # Собираем все пути к наблюдениям
        self.observations = []
        for planet_folder in self.planet_folders:
            # Находим все папки FGS1_calibration_{i}
            fgs_folders = [
                planet_folder/fgs_folder
                for fgs_folder in os.listdir(planet_folder)
                if fgs_folder.startswith('FGS1_calibration_')
            ]
            
            # Для каждой папки FGS добавляем пути к наблюдениям
            for fgs_folder in fgs_folders:
                # Извлекаем индекс наблюдения из имени папки FGS1_calibration_{i}
                observation_index = fgs_folder.name.split('_')[-1]
                planet_id = os.path.basename(planet_folder)
                
                self.observations.append({
                    'path': planet_folder,  # Путь к папке планеты
                    'planet_id': planet_id,
                    'observation_index': observation_index  # Индекс наблюдения
                })

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # Получаем путь к папке наблюдения
        observation = self.observations[idx]
        
        # Вызываем функцию получения сигнала
        signal = self.get_signal(observation)
        try:
            labels = self.get_labels_for_planet(observation['planet_id'])
            noise = get_noise(signal, labels[1:])
        except ValueError:
            labels = None
            noise = None
        
        return {
            'signal': signal,
            'noise' : noise,
            'noisy_target' : get_target(signal),
            'target': labels,

            'path': observation['path'],
            'planet_id': observation['planet_id'],
            'observation_index': observation['observation_index']
        }

    def get_signal(self, observation):
        # TODO: Реализуйте логику загрузки сигнала 
        p = observation['path']
        i = observation['observation_index']

        signal_airs = pd.read_parquet(p / f'AIRS-CH0_signal_{i}.parquet').values
        signal_airs = signal_airs.reshape(11250, 32, 356)
        # signal_fgs1 = pd.read_parquet(p / f'FGS1_signal_{i}.parquet').values
        # signal_fgs1 = signal_fgs1.reshape(135000, 32, 32)
        
        calibration_dir = p / f'AIRS-CH0_calibration_{i}'
        dark = pd.read_parquet(calibration_dir / 'dark.parquet').values
        dead = pd.read_parquet(calibration_dir / 'dead.parquet').values
        linear_corr = pd.read_parquet(calibration_dir / 'linear_corr.parquet').values.reshape(6, 32, -1)
        flat = pd.read_parquet(calibration_dir / 'flat.parquet').values
        dt = self.axis_info['AIRS-CH0-integration_time'].dropna().values
        dt[1::2] += 0.1
        
        # adc_offset_fgs1 = self.adc_info['FGS1_adc_offset'].values[0]
        adc_offset_airs = self.adc_info['AIRS-CH0_adc_offset'].values[0]
        # adc_gain_fgs1 = self.adc_info['FGS1_adc_gain'].values[0]
        adc_gain_airs = self.adc_info['AIRS-CH0_adc_gain'].values[0]

        cds_signal = prepare_signal(signal_airs, 
                                    dead, dark, linear_corr=None, dt=dt, flat=flat,
                                    gain=adc_gain_airs, offset=adc_offset_airs,
                                    binning=30,
                                    calibrate=True)        # l, r = get_transit_bounds(cds_signal.mean(axis=(1,2)))
        # cds_signal = cds_signal[:,8:-8,l:r]
        cds_signal = cds_signal[:,8:-8,:]
        return cds_signal

    def get_labels_for_planet(self, planet_id):
        # Находим строку с соответствующим planet_id
        planet_labels = self.train_labels[
            self.train_labels['planet_id'] == int(planet_id)
        ]
        if len(planet_labels) == 0:
            raise ValueError(f"No labels found for planet_id {planet_id}")
        
        labels = planet_labels.drop(columns=['planet_id']).values[0]
        return labels


def collate_fn(batch):
    collated_batch = {}
    
    for key in batch[0].keys():
        if key == 'signal':
            collated_batch[key] = torch.stack([
                torch.tensor(item[key]) for item in batch])
            # collated_batch[key] = [torch.tensor(item[key]) for item in batch]
        elif key == 'target':
            try:
                collated_batch[key] = torch.stack([
                    torch.tensor(item[key]) for item in batch])
                # collated_batch[key] = [torch.tensor(item[key]) for item in batch]
            except:
                collated_batch[key] = [
                    torch.tensor(item[key]) for item in batch]
                # pass
    
    return collated_batch

def create_planet_dataloaders(dataset, val_split=0.2, batch_size=32, num_workers=4):
    """
    Создает train и validation DataLoader с стратифицированным разбиением
    
    Args:
        dataset (PlanetDataset): Исходный датасет
        val_split (float): Доля validation выборки
        batch_size (int): Размер батча
        num_workers (int): Количество воркеров для загрузки данных
    
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # Извлекаем planet_id для стратификации
    planet_ids = [obs['planet_id'] for obs in dataset.observations]
    
    # Создаем индексы для train и validation с сохранением стратификации по planet_id
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=val_split, 
        random_state=42  # Для воспроизводимости
    )
    
    # Создаем подмножества датасета
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Создаем DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Для validation не нужно перемешивание 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader

def save_preprocessed_data(dataset, save_dir):
    """
    Сохраняет предобработанные данные в виде тензоров
    """
    save_dir = Path(save_dir)
    signals_dir       = save_dir / 'signals'
    noises_dir        = save_dir / 'noises'
    noisy_targets_dir = save_dir / 'noisy_targets'
    targets_dir       = save_dir / 'targets'
    # create directories for files
    for directory in [signals_dir, noises_dir, noisy_targets_dir, targets_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    dirs = {'signal'        : signals_dir, 
            'noise'         : noises_dir, 
            'noisy_target' : noisy_targets_dir, 
            'targets_dir'   : targets_dir}
    
    metadata = []
    
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        
        # Сохраняем сигнал (4D тензор)
        for key in item:
            if key in ['signal', 'noise', 'noisy_target']:
                torch.save(item[key], dirs[key] / f"{idx}.pt")
        # Сохраняем target если он есть
        target = item['target']
        if target is not None:
            torch.save(torch.tensor(target), targets_dir / f"{idx}.pt")
        
        # Сохраняем метаданные
        metadata.append({
            'index': idx,
            'planet_id': item['planet_id'],
            'observation_index': item['observation_index'],
            'has_target': target is not None
        })
    
    # Сохраняем метаданные
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(save_dir / 'metadata.csv', index=False)

class PreprocessedPlanetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(self.data_dir / 'metadata.csv')
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Загружаем сигнал
        res = {}
        for key in ['signal', 'noise', 'noisy_target']:
            res[key] = torch.load(self.data_dir / f'{key}s' / f"{row['index']}.pt", weights_only=False)
        # Загружаем target если он есть
        target = None
        if row['has_target']:
            res['target'] = torch.load(self.data_dir / 'targets' / f"{row['index']}.pt", 
                                     weights_only=False)
        res['planet_id']         = row['planet_id']
        res['observation_index'] = row['observation_index']
        
        return res

# Создайте соответствующий collate_fn
def preprocessed_collate_fn(batch):
    signals       = torch.stack([torch.tensor(item['signal'      ]) for item in batch])
    noises        = torch.stack([torch.tensor(item['noise'       ]) for item in batch])
    noisy_targets = torch.stack([torch.tensor(item['noisy_target']) for item in batch])
    targets = [item['target'] for item in batch]
    
    # Фильтруем None targets
    valid_targets = [t for t in targets if t is not None]
    if len(valid_targets) == len(targets):
        targets = torch.stack(valid_targets)
    else:
        targets = None
    
    return {
        'signal'      : signals,
        'noise'       : noises,
        'noisy_target': noisy_targets,
        'target'      : targets,
        'planet_id': [item['planet_id'] for item in batch],
        'observation_index': [item['observation_index'] for item in batch]
    }

def split_dataset(dataset, val_fraction=0.2, seed=42):
    """
    Return (train_dataset, val_dataset) as torch.utils.data.Subset objects.
    """
    from torch.utils.data import Subset
    import torch
    import math
    num_samples = len(dataset)
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed)).tolist()
    val_size = int(math.ceil(val_fraction * num_samples))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


