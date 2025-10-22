import os
import dgl
import torch
import random
import numpy as np

from parse_args import args
from model import DrugRepositioningModel
from hypergraph_data import HypergraphDatasetDGL
from dataloader import DrugDiseaseKFoldDataset
from model_train import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)

def train_and_evaluate(device, log_folder_path):
    hypergraphDataset = HypergraphDatasetDGL(device = device)
    hypergraph = hypergraphDataset.g
    incidence, D_v_neg_sqrt, D_e_neg = hypergraphDataset.get_hypergraph_matrices()
    num_drugs = hypergraphDataset.num_drugs
    num_diseases = hypergraphDataset.num_diseases

    kfold_dataset = DrugDiseaseKFoldDataset(hypergraphDataset)

    l_test_fold_results = []
    for train_loader, val_loader, test_loader, fold_i in kfold_dataset.get_fold_dataloaders():
        print(f"Training Fold {fold_i}")
        
        model = DrugRepositioningModel(hypergraph, incidence, D_v_neg_sqrt, D_e_neg, num_drugs, num_diseases, device).to(device)

        trainer = Trainer(model, device, log_folder_path)

        trainer.train(train_loader, val_loader, fold_i)

        out_path = os.path.join(log_folder_path, "fold_{}".format(fold_i))
        ckpt_path = os.path.join(out_path, "result.ckpt")
        trainer.model.load(ckpt_path)

        test_results_fold_i = trainer.evaluate(test_loader)

        l_test_fold_results.append(test_results_fold_i)
    
    np_results = np.array(l_test_fold_results)
    mean_result = np.round(np.mean(np_results, axis = 0), 4)
    print(mean_result)

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    if use_cuda and args.gpu < torch.cuda.device_count():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    set_seed(args.seed)

    current_dir = os.getcwd()
    log_folder_path = os.path.join(current_dir, "Outputs/DDA/{}".format(args.dataset))
    
    train_and_evaluate(device, log_folder_path)