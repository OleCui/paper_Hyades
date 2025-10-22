import os
import dgl
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from hypergraph_data import HypergraphDatasetDGL
from parse_args import args

class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, input_dropout, hidden_dropout, feature_map_dropout, channels, kernel_size, use_bias=True):
        super(ConvE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.embedding_height = 16
        self.embedding_width = embedding_dim // self.embedding_height
        assert self.embedding_height * self.embedding_width == embedding_dim
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.feature_map_dropout = nn.Dropout2d(feature_map_dropout)
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=channels,
            kernel_size=(kernel_size, kernel_size),
            stride=1,
            padding=1,
            bias=use_bias
        )
        
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
        input_height = 2 * self.embedding_height
        input_width = self.embedding_width
        
        padding = 1
        stride = 1
        
        conv_height = ((input_height + 2 * padding - kernel_size) // stride) + 1
        conv_width = ((input_width + 2 * padding - kernel_size) // stride) + 1
        
        self.flattened_size = channels * conv_height * conv_width
        
        self.fc = nn.Linear(self.flattened_size, embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        
    def forward(self, head_idx, relation_idx, tail_idx=None):
        batch_size = head_idx.shape[0]
        
        head_emb = self.entity_embeddings(head_idx)
        relation_emb = self.relation_embeddings(relation_idx)
        
        head_emb = head_emb.view(batch_size, 1, self.embedding_height, self.embedding_width)
        relation_emb = relation_emb.view(batch_size, 1, self.embedding_height, self.embedding_width)
        
        stacked = torch.cat([head_emb, relation_emb], dim=2)
        stacked = self.bn0(stacked)
        stacked = self.input_dropout(stacked)
        
        x = self.conv1(stacked)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        if tail_idx is not None:
            tail_emb = self.entity_embeddings(tail_idx)
            scores = torch.sum(x * tail_emb, dim=1)
        else:
            scores = torch.mm(x, self.entity_embeddings.weight.t())
            
        return scores
    
    def get_embeddings(self):
        return self.entity_embeddings.weight.detach()


class KGDataset(Dataset):
    def __init__(self, triples, num_entities, num_relations, negative_ratio=1):
        self.triples = triples
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.negative_ratio = negative_ratio
    
        self.true_triples = set(triples)
        
        self.use_cached_negatives = negative_ratio > 10
        if self.use_cached_negatives:
            self._precompute_negatives()
    
    def _precompute_negatives(self):
        self.cached_negatives = []
        
        for h, r, t in tqdm(self.triples, desc='Generating negative samples'):
            negatives = []
            tried = set()
            
            while len(negatives) < self.negative_ratio:
                neg_t = np.random.randint(0, self.num_entities)
                
                if neg_t not in tried:
                    tried.add(neg_t)
                    if (h, r, neg_t) not in self.true_triples:
                        negatives.append(neg_t)
                
                if len(tried) >= self.num_entities:
                    remaining = self.negative_ratio - len(negatives)
                    random_negs = np.random.randint(0, self.num_entities, remaining)
                    negatives.extend(random_negs.tolist())
                    break
            
            self.cached_negatives.append(negatives[:self.negative_ratio])
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        if self.negative_ratio == 0:
            return {
                'head': h,
                'relation': r,
                'tail': t,
                'negative_tails': []
            }
        
        if self.use_cached_negatives:
            negative_samples = self.cached_negatives[idx]
        else:
            negative_samples = []
            tried = 0
            for _ in range(self.negative_ratio):
                neg_t = np.random.randint(0, self.num_entities)
                while (h, r, neg_t) in self.true_triples and tried < 100:
                    neg_t = np.random.randint(0, self.num_entities)
                    tried += 1
                negative_samples.append(neg_t)
        
        return {
            'head': h,
            'relation': r,
            'tail': t,
            'negative_tails': negative_samples
        }


def collate_kg_batch(batch):
    heads = torch.LongTensor([item['head'] for item in batch])
    relations = torch.LongTensor([item['relation'] for item in batch])
    tails = torch.LongTensor([item['tail'] for item in batch])
    
    if len(batch[0]['negative_tails']) > 0:
        neg_tails = torch.LongTensor([item['negative_tails'] for item in batch])
    else:
        neg_tails = torch.empty(0, dtype=torch.long)
    
    return {
        'head': heads,
        'relation': relations,
        'tail': tails,
        'negative_tails': neg_tails
    }

class ConvETrainer:
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-5, label_smoothing=0.1):
        self.model = model.to(device)
        self.device = device
        self.label_smoothing = label_smoothing

        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc='Training'):
            heads = batch['head'].to(self.device)
            relations = batch['relation'].to(self.device)
            tails = batch['tail'].to(self.device)
            neg_tails = batch['negative_tails'].to(self.device)
            
            if neg_tails.numel() == 0:
                continue
            
            batch_size = heads.shape[0]
            negative_ratio = neg_tails.shape[1]
            
            all_heads = heads.repeat(negative_ratio + 1)
            all_relations = relations.repeat(negative_ratio + 1)
            
            all_tails = torch.cat([
                tails.unsqueeze(1), 
                neg_tails  
            ], dim=1).view(-1)
            
            all_scores = self.model(all_heads, all_relations, all_tails)
            
            labels = torch.zeros(batch_size * (negative_ratio + 1)).to(self.device)
            labels[::negative_ratio + 1] = 1
        
            if self.label_smoothing > 0:
                labels = labels * (1 - self.label_smoothing) + self.label_smoothing / 2
            
            loss = self.criterion(all_scores, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def train_epoch_1n_scoring(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc='Training (1-N scoring)'):
            heads = batch['head'].to(self.device)
            relations = batch['relation'].to(self.device)
            tails = batch['tail'].to(self.device)
            
            batch_size = heads.shape[0]
            all_scores = self.model(heads, relations, tail_idx=None)
            
            targets = torch.zeros(batch_size, self.model.num_entities).to(self.device)
            
            for i, tail in enumerate(tails):
                targets[i, tail] = 1.0
            
            if self.label_smoothing > 0:
                targets = targets * (1 - self.label_smoothing) + \
                         self.label_smoothing / self.model.num_entities
            
            loss = F.binary_cross_entropy_with_logits(all_scores, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        
        ranks = []
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_10 = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                heads = batch['head'].to(self.device)
                relations = batch['relation'].to(self.device)
                tails = batch['tail'].to(self.device)
                
                scores = self.model(heads, relations)
                
                for i, (h, r, t) in enumerate(zip(heads, relations, tails)):
                    target_score = scores[i, t].item()
                    
                    rank = torch.sum(scores[i] > target_score).item() + 1
                    ranks.append(rank)
                    
                    if rank <= 1:
                        hits_at_1 += 1
                    if rank <= 3:
                        hits_at_3 += 1
                    if rank <= 10:
                        hits_at_10 += 1
        
        num_samples = len(ranks)
        mrr = np.mean([1/r for r in ranks])
        mr = np.mean(ranks)
        hits_at_1 = hits_at_1 / num_samples
        hits_at_3 = hits_at_3 / num_samples
        hits_at_10 = hits_at_10 / num_samples
        
        return {
            'MRR': mrr,
            'MR': mr,
            'Hits@1': hits_at_1,
            'Hits@3': hits_at_3,
            'Hits@10': hits_at_10
        }

def prepare_kg_data(hypergraph_dataset):
    g = hypergraph_dataset.g
    triples = []
    entity_to_type = {}
    
    drug_offset = 0
    disease_offset = hypergraph_dataset.num_drugs
    gene_offset = disease_offset + hypergraph_dataset.num_diseases
    
    num_entities = (hypergraph_dataset.num_drugs + 
                   hypergraph_dataset.num_diseases +
                   hypergraph_dataset.num_genes)
    
    for i in range(hypergraph_dataset.num_drugs):
        entity_to_type[i] = 'drug'
    for i in range(hypergraph_dataset.num_diseases):
        entity_to_type[disease_offset + i] = 'disease'
    for i in range(hypergraph_dataset.num_genes):
        entity_to_type[gene_offset + i] = 'gene'
    
    relation_types = {}
    relation_id = 0
    
    if ('drug', 'drug-gene', 'gene') in g.canonical_etypes:
        src, dst = g.edges(etype=('drug', 'drug-gene', 'gene'))
        if 'drug-gene' not in relation_types:
            relation_types['drug-gene'] = relation_id
            relation_id += 1
        
        for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
            triples.append((
                s + drug_offset,
                relation_types['drug-gene'],
                d + gene_offset
            ))
    
    if ('disease', 'disease-gene', 'gene') in g.canonical_etypes:
        src, dst = g.edges(etype=('disease', 'disease-gene', 'gene'))
        if 'disease-gene' not in relation_types:
            relation_types['disease-gene'] = relation_id
            relation_id += 1
        
        for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
            triples.append((
                s + disease_offset,
                relation_types['disease-gene'],
                d + gene_offset
            ))
    
    reverse_triples = []
    for h, r, t in triples:
        reverse_r = r + relation_id
        reverse_triples.append((t, reverse_r, h))
    
    triples.extend(reverse_triples)
    num_relations = relation_id * 2
    
    return triples, num_entities, num_relations, entity_to_type


def main(USE_1N_SCORING = True):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hypergraph_dataset = HypergraphDatasetDGL(device=device)
    
    triples, num_entities, num_relations, entity_to_type = prepare_kg_data(hypergraph_dataset)
    
    random.shuffle(triples)
    split_idx = int(len(triples) * 0.85)
    train_triples = triples[:split_idx]
    valid_triples = triples[split_idx:]
    
    if USE_1N_SCORING:
        negative_ratio = 0
        batch_size = 512
    else:
        negative_ratio = 5
        batch_size = 256
    
    train_dataset = KGDataset(
        train_triples, 
        num_entities, 
        num_relations,
        negative_ratio=negative_ratio
    )
    
    valid_dataset = KGDataset(
        valid_triples,
        num_entities,
        num_relations,
        negative_ratio=4
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_kg_batch if not USE_1N_SCORING else None
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_kg_batch
    )
    
    model = ConvE(
        num_entities = num_entities,
        num_relations = num_relations,
        embedding_dim = args.in_dim,
        input_dropout = 0.2,
        hidden_dropout = 0.3,
        feature_map_dropout = 0.2,
        channels = args.convE_channels,
        kernel_size = args.convE_kernel_size
    )
    
    trainer = ConvETrainer(
        model=model,
        device=device,
        learning_rate= args.lr,
        weight_decay=1e-5,
        label_smoothing=0.1 
    )
    
    num_epochs = 2000
    patience = 100
    best_mrr = 0
    patience_counter = 0

    current_dir = os.getcwd()
    model_checkpoint_path = os.path.join(current_dir, "Outputs/ConvE/{}".format(args.dataset))

    if not os.path.exists(model_checkpoint_path):
        os.makedirs(model_checkpoint_path)

    for epoch in range(num_epochs):
        if USE_1N_SCORING:
            train_loss = trainer.train_epoch_1n_scoring(train_loader)
        else:
            train_loss = trainer.train_epoch(train_loader)
            
        if (epoch + 1) % 1 == 0:
            metrics = trainer.evaluate(valid_loader)
            
            if metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                patience_counter = 0
                
                torch.save(model.state_dict(), os.path.join(model_checkpoint_path, "best_conve_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    model.load_state_dict(torch.load(os.path.join(model_checkpoint_path, "best_conve_model.pth")))
    
    embeddings = model.get_embeddings().cpu()

    drug_embeddings = embeddings[:hypergraph_dataset.num_drugs]
    disease_embeddings = embeddings[
        hypergraph_dataset.num_drugs:
        hypergraph_dataset.num_drugs + hypergraph_dataset.num_diseases
    ]
    gene_embeddings = embeddings[
        hypergraph_dataset.num_drugs + hypergraph_dataset.num_diseases:
        hypergraph_dataset.num_drugs + hypergraph_dataset.num_diseases + hypergraph_dataset.num_genes
    ]

    torch.save(drug_embeddings, os.path.join(model_checkpoint_path, "drug_embeddings.pt"))
    torch.save(disease_embeddings, os.path.join(model_checkpoint_path, "disease_embeddings.pt"))
    torch.save(gene_embeddings, os.path.join(model_checkpoint_path, "gene_embeddings.pt"))

if __name__ == "__main__":
    main()