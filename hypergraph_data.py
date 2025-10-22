import os
import dgl
import json
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from collections import defaultdict
from parse_args import args
from sklearn.model_selection import StratifiedKFold, train_test_split

class HypergraphDatasetDGL:
    def __init__(self, device):
        self.dataset = args.dataset
        self.seed = args.seed
        self.negative_ratio = args.negative_rate
        self.K_fold = args.K_fold
        self.validation_ratio = args.validation_ratio
        self.device = device

        self.load_data()
        
        self.build_dgl_hypergraph()

        self.extract_hypergraph_matrices()

        self.build_training_dataset()
    
    def load_data(self):
        current_dir = os.getcwd()
        data_path = os.path.join(current_dir, "Datasets/{}".format(self.dataset))

        with open(os.path.join(data_path, "drug_to_idx.json"), 'r') as f:
            self.d_drug_name2id = json.load(f)
        
        with open(os.path.join(data_path, "disease_to_idx.json"), 'r') as f:
            self.d_disease_name2id = json.load(f)

        with open(os.path.join(data_path, "gene_to_idx.json"), 'r') as f:
            self.d_gene_name2id = json.load(f)

        self.d_drug_id2name = {idx: drug for drug, idx in self.d_drug_name2id.items()}
        self.d_disease_id2name = {idx: disease for disease, idx in self.d_disease_name2id.items()}
        self.d_gene_id2name = {idx: gene for gene, idx in self.d_gene_name2id.items()}

        self.num_drugs = len(self.d_drug_name2id)
        self.num_diseases = len(self.d_disease_name2id)
        self.num_genes = len(self.d_gene_name2id)

        self.df_DDAs = pd.read_csv(os.path.join(data_path, "drug-disease.csv"))
        self.df_drug_genes = pd.read_csv(os.path.join(data_path, "drug-gene.csv"))
        self.df_disease_genes = pd.read_csv(os.path.join(data_path, "disease-gene.csv"))

        self.drug_disease_pairs = [(int(row['drug_idx']), int(row['disease_idx'])) for _, row in self.df_DDAs.iterrows()]

        self.positive_pairs_set = set(self.drug_disease_pairs)

    def build_dgl_hypergraph(self):
        gene_to_drugs = defaultdict(set)
        gene_to_diseases = defaultdict(set)

        for _, row in self.df_drug_genes.iterrows():
            drug_idx = int(row['drug_idx'])
            gene_idx = int(row['gene_idx'])
            gene_to_drugs[gene_idx].add(drug_idx)
        
        for _, row in self.df_disease_genes.iterrows():
            disease_idx = int(row['disease_idx'])
            gene_idx = int(row['gene_idx'])
            gene_to_diseases[gene_idx].add(disease_idx)
        
        drug_gene_edges = []
        disease_gene_edges = []
        gene_drug_edges = []
        gene_disease_edges = []

        for gene_idx in range(self.num_genes):
            for drug_idx in gene_to_drugs.get(gene_idx, []):
                drug_gene_edges.append((drug_idx, gene_idx))
                gene_drug_edges.append((gene_idx, drug_idx))
                
            for disease_idx in gene_to_diseases.get(gene_idx, []):
                disease_gene_edges.append((disease_idx, gene_idx))
                gene_disease_edges.append((gene_idx, disease_idx))
        
        graph_data = {
            ('drug', 'drug-gene', 'gene'): drug_gene_edges,
            ('disease', 'disease-gene', 'gene'): disease_gene_edges,
            
            ('gene', 'gene-drug', 'drug'): gene_drug_edges,
            ('gene', 'gene-disease', 'disease'): gene_disease_edges,
        }
        
        self.g = dgl.heterograph(graph_data)
        
        self.g = self.g.to(self.device)
        
    def extract_hypergraph_matrices(self) -> None:
        num_nodes = self.num_drugs + self.num_diseases
        num_hyperedges = self.num_genes
        
        self.H = torch.zeros(num_nodes, num_hyperedges, device=self.device)

        if 'drug-gene' in self.g.etypes:
            drug_gene_edges = self.g['drug-gene'].edges()
            drug_indices = drug_gene_edges[0].cpu().numpy()
            gene_indices = drug_gene_edges[1].cpu().numpy()
            
            for drug_idx, gene_idx in zip(drug_indices, gene_indices):
                self.H[drug_idx, gene_idx] = 1.0
        
        if 'disease-gene' in self.g.etypes:
            disease_gene_edges = self.g['disease-gene'].edges()
            disease_indices = disease_gene_edges[0].cpu().numpy()
            gene_indices = disease_gene_edges[1].cpu().numpy()
            
            for disease_idx, gene_idx in zip(disease_indices, gene_indices):
                node_idx = disease_idx + self.num_drugs
                self.H[node_idx, gene_idx] = 1.0
    
        D_v, D_e = self._compute_degree_matrices(self.H)
        
        self.D_v_neg_sqrt = self._compute_matrix_power(D_v, -0.5)
        self.D_e_neg = self._compute_matrix_power(D_e, -1.0)
        
    def _compute_degree_matrices(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        node_degrees = torch.sum(H, dim=1)  
        hyperedge_degrees = torch.sum(H, dim=0)
        D_v = torch.diag(node_degrees)
        D_e = torch.diag(hyperedge_degrees)
        
        return D_v, D_e
    
    def _compute_matrix_power(self, diagonal_matrix: torch.Tensor, power: float, eps: float = 1e-10) -> torch.Tensor:
        diagonal_values = torch.diagonal(diagonal_matrix)
        
        diagonal_values = torch.where(diagonal_values > eps, diagonal_values, torch.full_like(diagonal_values, eps))
        
        powered_values = torch.pow(diagonal_values, power)
        
        powered_matrix = torch.diag(powered_values)
        
        return powered_matrix

    def get_hypergraph_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.H, self.D_v_neg_sqrt, self.D_e_neg

    def _sample_negative_pairs_with_count(self, num_negatives):
        negative_pairs = self._random_negative_sampling(num_negatives)
        
        return negative_pairs
        
    def _random_negative_sampling(self, num_negatives):
        negative_pairs_set = set()
        max_attempts = num_negatives * 10
        attempts = 0
        
        while len(negative_pairs_set) < num_negatives and attempts < max_attempts:
            drug_idx = np.random.randint(0, self.num_drugs)
            disease_idx = np.random.randint(0, self.num_diseases)
            attempts += 1
            
            candidate_pair = (drug_idx, disease_idx)
            if candidate_pair not in self.positive_pairs_set:
                negative_pairs_set.add(candidate_pair)
        
        return list(negative_pairs_set)
    
    def build_training_dataset(self):
        used_positive_pairs = self.drug_disease_pairs
        num_negatives = int(len(used_positive_pairs) * self.negative_ratio)
        negative_pairs = self._sample_negative_pairs_with_count(num_negatives)

        all_pairs = used_positive_pairs + negative_pairs
        all_labels = [1] * len(used_positive_pairs) + [0] * len(negative_pairs)

        self.np_all_pairs = np.array(all_pairs)
        self.np_all_labels = np.array(all_labels)
    
    def get_stratified_kfold_splits(self, shuffle=True):
        skf = StratifiedKFold(n_splits = self.K_fold, shuffle=shuffle, random_state = self.seed)
        
        fold_idx = 0
        for train_idx, test_idx in skf.split(self.np_all_pairs, self.np_all_labels):
            fold_idx += 1

            test_pairs = self.np_all_pairs[test_idx]
            test_labels = self.np_all_labels[test_idx]

            temp_train_pairs = self.np_all_pairs[train_idx]
            temp_train_labels = self.np_all_labels[train_idx]

            train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            temp_train_pairs, temp_train_labels, test_size = self.validation_ratio, stratify = temp_train_labels, random_state = self.seed + fold_idx)
            
            yield train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels