import torch
from parse_args import args
from torch.utils.data import Dataset, DataLoader

class DrugDiseaseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = torch.LongTensor(pairs)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        sample = {
            'drug_id': self.pairs[idx, 0],
            'disease_id': self.pairs[idx, 1], 
            'label': self.labels[idx]}
            
        return sample

class DrugDiseaseKFoldDataset:
    def __init__(self, hypergraphDataset, shuffle=True, num_workers=4):
        self.hypergraphDataset = hypergraphDataset
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def get_fold_dataloaders(self):
        for fold_idx, (train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels) in enumerate(
            self.hypergraphDataset.get_stratified_kfold_splits()):
            train_dataset = DrugDiseaseDataset(train_pairs, train_labels)
            valid_dataset = DrugDiseaseDataset(val_pairs, val_labels)
            test_dataset = DrugDiseaseDataset(test_pairs, test_labels)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=True)
            
            val_loader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            
            yield train_loader, val_loader, test_loader, fold_idx + 1