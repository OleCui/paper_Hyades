import os
import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from typing import Dict

from parse_args import args


class BahdanauAttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttentionFusion, self).__init__()
        
        self.W_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=True)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.xavier_uniform_(self.W_2.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.zeros_(self.W_1.bias)
        nn.init.zeros_(self.W_2.bias)
        nn.init.zeros_(self.v.bias)
    
    def forward(self, message1, message2):
        proj_msg1 = self.W_1(message1)  
        proj_msg2 = self.W_2(message2)  
        
        energy1 = self.v(torch.tanh(proj_msg1)).squeeze(-1)
        energy2 = self.v(torch.tanh(proj_msg2)).squeeze(-1)
        
        energies = torch.stack([energy1, energy2], dim=1)
        attention_weights = F.softmax(energies, dim=1)
        
        fused_message = (attention_weights[:, 0:1] * message1 + 
                        attention_weights[:, 1:2] * message2)
        
        return fused_message

class NodeToHyperedgeLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(NodeToHyperedgeLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        
        self.W_Q = nn.Linear(in_dim, out_dim)
        self.W_K = nn.Linear(in_dim, out_dim)
        self.W_V = nn.Linear(in_dim, out_dim)

        self._init_parameters()
    
    def _init_parameters(self):
        for linear in [self.W_Q, self.W_K, self.W_V]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        
    def forward(self, g, feat):
        with g.local_scope():
            if isinstance(feat, tuple):
                h_src, h_dst = feat
            else:
                h_src = feat
                h_dst = feat

            Q = self.W_Q(h_dst).view(-1, self.num_heads, self.out_dim // self.num_heads)
            K = self.W_K(h_src).view(-1, self.num_heads, self.out_dim // self.num_heads)
            V = self.W_V(h_src).view(-1, self.num_heads, self.out_dim // self.num_heads)
            
            g.dstdata['Q'] = Q
            g.srcdata['K'] = K
            g.srcdata['V'] = V
            
            g.apply_edges(fn.u_dot_v('K', 'Q', 'score'))
            g.edata['score'] = g.edata['score'] / np.sqrt(self.out_dim // self.num_heads)
            
            g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['score'])
            
            g.update_all(
                fn.u_mul_e('V', 'a', 'm'),
                fn.sum('m', 'h'))
            
            h_out = g.dstdata['h'].view(-1, self.out_dim)
            
            return h_out

class HyperedgeToNodeLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(HyperedgeToNodeLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        
        self.W_Q = nn.Linear(in_dim, out_dim)
        self.W_K = nn.Linear(in_dim, out_dim)
        self.W_V = nn.Linear(in_dim, out_dim)
        self._init_parameters()
    
    def _init_parameters(self):
        for linear in [self.W_Q, self.W_K, self.W_V]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        
    def forward(self, g, feat):
        with g.local_scope():
            if isinstance(feat, tuple):
                h_src, h_dst = feat
            else:
                h_src = feat
                h_dst = feat

            Q = self.W_Q(h_dst).view(-1, self.num_heads, self.out_dim // self.num_heads)
            K = self.W_K(h_src).view(-1, self.num_heads, self.out_dim // self.num_heads)
            V = self.W_V(h_src).view(-1, self.num_heads, self.out_dim // self.num_heads)
            
            g.dstdata['Q'] = Q
            g.srcdata['K'] = K
            g.srcdata['V'] = V
            
            g.apply_edges(fn.u_dot_v('K', 'Q', 'score'))
            g.edata['score'] = g.edata['score'] / np.sqrt(self.out_dim // self.num_heads)
            g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['score'])
            
            g.update_all(
                fn.u_mul_e('V', 'a', 'm'),
                fn.sum('m', 'h'))
            
            h_out = g.dstdata['h'].view(-1, self.out_dim)
            
            return h_out

class HGNNLayerDGL(nn.Module):
    def __init__(self, device, in_dim, out_dim, num_heads, dropout):
        super(HGNNLayerDGL, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.node_to_edge_layers = nn.ModuleDict({
            'drug-gene': NodeToHyperedgeLayer(in_dim, out_dim, num_heads),
            'disease-gene': NodeToHyperedgeLayer(in_dim, out_dim, num_heads),
        })
        
        self.edge_to_node_layers = nn.ModuleDict({
            'gene-drug': HyperedgeToNodeLayer(out_dim, out_dim, num_heads),
            'gene-disease': HyperedgeToNodeLayer(out_dim, out_dim, num_heads),
        })
        
        self.bahdanau_fusion_gene = BahdanauAttentionFusion(out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

    def forward(self, g, node_feats):
        edge_feats = {}
        
        if 'gene' in node_feats:
            num_genes = node_feats['gene'].shape[0]
            gene_messages_from_drug = torch.zeros(num_genes, self.out_dim).to(self.device)
            gene_messages_from_disease = torch.zeros(num_genes, self.out_dim).to(self.device)
            
            has_drug_connection = torch.zeros(num_genes, dtype=torch.bool).to(self.device)
            has_disease_connection = torch.zeros(num_genes, dtype=torch.bool).to(self.device)
            
            if 'drug-gene' in g.etypes and g.num_edges('drug-gene') > 0:
                subg = g['drug-gene']
                _, gene_indices = subg.edges()
                unique_gene_indices = torch.unique(gene_indices)
                has_drug_connection[unique_gene_indices] = True
                
                gene_msg = self.node_to_edge_layers['drug-gene'](
                    subg, (node_feats['drug'], node_feats['gene']))
                
                if gene_msg is not None:
                    gene_messages_from_drug = gene_msg
            
            if 'disease-gene' in g.etypes and g.num_edges('disease-gene') > 0:
                subg = g['disease-gene']
                _, gene_indices = subg.edges()
                unique_gene_indices = torch.unique(gene_indices)
                has_disease_connection[unique_gene_indices] = True
                
                gene_msg = self.node_to_edge_layers['disease-gene'](
                    subg, (node_feats['disease'], node_feats['gene']))
                
                if gene_msg is not None:
                    gene_messages_from_disease = gene_msg
            
            gene_final_messages = torch.zeros(num_genes, self.out_dim).to(self.device)
            
            both_connected = has_drug_connection & has_disease_connection
            if both_connected.any():
                gene_final_messages[both_connected] = self.bahdanau_fusion_gene(
                    gene_messages_from_drug[both_connected],
                    gene_messages_from_disease[both_connected])
            
            only_drug = has_drug_connection & ~has_disease_connection
            if only_drug.any():
                gene_final_messages[only_drug] = gene_messages_from_drug[only_drug]
            
            only_disease = ~has_drug_connection & has_disease_connection
            if only_disease.any():
                gene_final_messages[only_disease] = gene_messages_from_disease[only_disease]
            
            edge_feats['gene'] = gene_final_messages

        if 'gene' in edge_feats and 'gene' in node_feats:
            edge_feats['gene'] = self.layer_norm1(
                node_feats['gene'] + self.dropout(F.gelu(edge_feats['gene'])))
        
        edge_feats['drug'] = node_feats['drug']
        edge_feats['disease'] = node_feats['disease']
        
        new_node_feats = {}
        
        if 'drug' in node_feats:
            num_drugs = node_feats['drug'].shape[0]
            drug_messages = torch.zeros(num_drugs, self.out_dim).to(self.device)
            
            if 'gene-drug' in g.etypes and g.num_edges('gene-drug') > 0:
                subg = g['gene-drug']
                drug_msg = self.edge_to_node_layers['gene-drug'](
                    subg, (edge_feats['gene'], edge_feats['drug']))
                
                if drug_msg is not None:
                    drug_messages = drug_msg
            
            new_node_feats['drug'] = drug_messages

        if 'disease' in node_feats:
            num_diseases = node_feats['disease'].shape[0]
            disease_messages = torch.zeros(num_diseases, self.out_dim).to(self.device)
            
            if 'gene-disease' in g.etypes and g.num_edges('gene-disease') > 0:
                subg = g['gene-disease']
                disease_msg = self.edge_to_node_layers['gene-disease'](
                    subg, (edge_feats['gene'], edge_feats['disease']))
                
                if disease_msg is not None:
                    disease_messages = disease_msg
            
            new_node_feats['disease'] = disease_messages

        if 'drug' in new_node_feats and 'drug' in node_feats:
            new_node_feats['drug'] = self.layer_norm2(
                node_feats['drug'] + self.dropout(F.gelu(new_node_feats['drug'])))

        if 'disease' in new_node_feats and 'disease' in node_feats:
            new_node_feats['disease'] = self.layer_norm2(
                node_feats['disease'] + self.dropout(F.gelu(new_node_feats['disease'])))
        
        new_node_feats['gene'] = edge_feats.get('gene', node_feats.get('gene'))
        
        return new_node_feats

class HGNNEncoder(nn.Module):
    
    def __init__(self, device, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout):
        super(HGNNEncoder, self).__init__()

        self.input_projs = nn.ModuleDict({
            'drug': nn.Linear(in_dim, hidden_dim),
            'disease': nn.Linear(in_dim, hidden_dim),
            'gene': nn.Linear(in_dim, hidden_dim),
        })

        self.layers = nn.ModuleList([
            HGNNLayerDGL(device, hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_projs = nn.ModuleDict({
            'drug': nn.Linear(hidden_dim, out_dim),
            'disease': nn.Linear(hidden_dim, out_dim),
        })

        self._init_parameters()
    
    def _init_parameters(self):
        for proj_dict in [self.input_projs, self.output_projs]:
            for proj in proj_dict.values():
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        
    def forward(self, g, use_masked_features):
        if use_masked_features:
            assert self.training
            h = {}
            for ntype in g.ntypes:
                if 'mask_feat' in g.nodes[ntype].data:
                    h[ntype] = self.input_projs[ntype](g.nodes[ntype].data['mask_feat'])
                else:
                    h[ntype] = self.input_projs[ntype](g.nodes[ntype].data['feat'])
        else:
            assert not self.training
            h = {ntype: self.input_projs[ntype](g.nodes[ntype].data['feat']) for ntype in g.ntypes}

        for layer in self.layers:
            h = layer(g, h)

        out_feats = {
            'drug': self.output_projs['drug'](h['drug']),
            'disease': self.output_projs['disease'](h['disease'])
        }
        
        return out_feats

class HypergraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super(HypergraphConvLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        self.linear_transform = nn.Linear(in_dim, out_dim, bias=True)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.linear_transform.weight)
        if self.linear_transform.bias is not None:
            nn.init.zeros_(self.linear_transform.bias)
    
    def forward(self, x: torch.Tensor, H: torch.Tensor, 
                D_v_neg_sqrt: torch.Tensor, D_e_neg: torch.Tensor) -> torch.Tensor:

        x_transformed = self.linear_transform(x)
        x_norm = torch.matmul(D_v_neg_sqrt, x_transformed)
        x_hyperedge = torch.matmul(H.t(), x_norm)
        x_hyperedge_norm = torch.matmul(D_e_neg, x_hyperedge)
        x_node = torch.matmul(H, x_hyperedge_norm)
        output = torch.matmul(D_v_neg_sqrt, x_node)
        output = self.dropout_layer(output)
        
        return output

class HGNNDecoder(nn.Module):
    def __init__(self, incidence, D_v_neg_sqrt, D_e_neg, num_drugs, num_diseases, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super(HGNNDecoder, self).__init__()

        self.H = incidence
        self.D_v_neg_sqrt = D_v_neg_sqrt
        self.D_e_neg = D_e_neg
        self.num_drugs = num_drugs
        self.num_diseases = num_diseases
        
        assert num_layers >= 2
        self.conv_layers = nn.ModuleList()
        
        self.conv_layers.append(
            HypergraphConvLayer(in_dim, hidden_dim, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                HypergraphConvLayer(hidden_dim, hidden_dim, dropout=dropout))
        
        self.conv_layers.append(
            HypergraphConvLayer(hidden_dim, out_dim, dropout=0.0))
    
    def forward(self, encoder_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        drug_features = encoder_output['drug']
        disease_features = encoder_output['disease']

        x = torch.cat([drug_features, disease_features], dim=0)

        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, self.H, self.D_v_neg_sqrt, self.D_e_neg)
            
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
        
        drug_embeddings = x[:self.num_drugs, :]
        disease_embeddings = x[self.num_drugs:, :]
        
        return {
            'drug': drug_embeddings,
            'disease': disease_embeddings
        }
    

class DrugDiseasePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(DrugDiseasePredictor, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def _multiple_operator(self, a, b):
        return a * b

    def _rotate_operator(self, a, b):
        a_re, a_im = a.chunk(2, dim=-1)
        b_re, b_im = b.chunk(2, dim=-1)
        message_re = a_re * b_re - a_im * b_im
        message_im = a_re * b_im + a_im * b_re
        message = torch.cat([message_re, message_im], dim=-1)
        return message
                
    def forward(self, drug_embeds, disease_embeds):
        m_result = self._multiple_operator(drug_embeds, disease_embeds)
        r_result = self._rotate_operator(drug_embeds, disease_embeds)
        combined = torch.cat([drug_embeds, disease_embeds, m_result, r_result], dim=1)

        scores = self.mlp(combined).squeeze(-1)
        
        return torch.sigmoid(scores)
    
class AdaptiveMaskRateScheduler:
    def __init__(self, min_rate, max_rate, total_epochs, mask_strategy, fixed_rate=None):
        self.min_rate = min_rate
        self.max_rate = max_rate
        assert self.min_rate < self.max_rate

        self.total_epochs = total_epochs
        self.mask_strategy = mask_strategy
        if self.mask_strategy == 'fixed':
            assert fixed_rate is not None
            assert min_rate <= fixed_rate <= max_rate
            self.fixed_rate = fixed_rate
    
    def get_mask_rate(self, epoch):
        if self.mask_strategy == 'fixed':
            return self.fixed_rate


class DrugRepositioningModel(nn.Module):
    def __init__(self, hypergraph, incidence, D_v_neg_sqrt, D_e_neg, num_drugs, num_diseases, device):
        super(DrugRepositioningModel, self).__init__()
        
        self.g = hypergraph
        self.incidence = incidence
        self.D_v_neg_sqrt = D_v_neg_sqrt
        self.D_e_neg = D_e_neg
        
        self.num_drugs = num_drugs
        self.num_diseases = num_diseases

        self.device = device
        self.in_dim = args.in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim

        self.mask_strategy = args.mask_strategy
        self.min_mask_rate = args.min_mask_rate
        self.max_mask_rate = args.max_mask_rate
        self.fixed_mask_rate = args.fixed_mask_rate

        self.num_heads = args.num_heads
        self.num_layers_encoder = args.num_layers_encoder
        self.dropout = args.dropout

        self.remask_rate = args.remask_rate
        self.remask_view = args.remask_view

        self.num_layers_decoder = args.num_layers_decoder

        self.scale_factor = args.scale_factor
        
        self.get_hypergraph_embeddings()

        self.encoder = HGNNEncoder(self.device, self.in_dim, self.hidden_dim, self.out_dim, self.num_heads, self.num_layers_encoder, self.dropout)
        
        self.decoder = HGNNDecoder(self.incidence, self.D_v_neg_sqrt, self.D_e_neg, self.num_drugs, self.num_diseases, self.out_dim, self.hidden_dim, self.in_dim, self.num_layers_decoder, self.dropout)
        
        self.predictor = DrugDiseasePredictor(self.out_dim, self.hidden_dim, self.dropout)
        
        self.mask_scheduler = AdaptiveMaskRateScheduler(self.min_mask_rate, self.max_mask_rate, args.epochs, self.mask_strategy, self.fixed_mask_rate)
        
        self.mask_token_drug = nn.Parameter(torch.randn(1, self.in_dim))
        self.mask_token_disease = nn.Parameter(torch.randn(1, self.in_dim))
        
        self.dmask_token_drug = nn.Parameter(torch.randn(1, self.out_dim))
        self.dmask_token_disease = nn.Parameter(torch.randn(1, self.out_dim))
        
        nn.init.xavier_normal_(self.mask_token_drug)
        nn.init.xavier_normal_(self.mask_token_disease)
        nn.init.xavier_normal_(self.dmask_token_drug)
        nn.init.xavier_normal_(self.dmask_token_disease)

    def get_hypergraph_embeddings(self):
        current_dir = os.getcwd()
        data_path = os.path.join(current_dir, "Outputs/ConvE/{}".format(args.dataset))

        drug_embeddings_path = os.path.join(data_path, "drug_embeddings.pt")
        disease_embeddings_path = os.path.join(data_path, "disease_embeddings.pt")
        gene_embeddings_path = os.path.join(data_path, "gene_embeddings.pt")

        assert os.path.exists(drug_embeddings_path)
        drug_embeddings = torch.load(drug_embeddings_path, map_location=self.device)
        
        assert os.path.exists(disease_embeddings_path)
        disease_embeddings = torch.load(disease_embeddings_path, map_location=self.device)
        
        assert os.path.exists(gene_embeddings_path)
        gene_embeddings = torch.load(gene_embeddings_path, map_location=self.device)
        
        if args.freeze_pretrained:
            self.drug_embeddings = nn.Parameter(drug_embeddings, requires_grad=False)
            self.disease_embeddings = nn.Parameter(disease_embeddings, requires_grad=False)
            self.gene_embeddings = nn.Parameter(gene_embeddings, requires_grad=False)
        else:
            self.drug_embeddings = nn.Parameter(drug_embeddings, requires_grad=True)
            self.disease_embeddings = nn.Parameter(disease_embeddings, requires_grad=True)
            self.gene_embeddings = nn.Parameter(gene_embeddings, requires_grad=True)
        
        self.g.nodes['drug'].data['feat'] = self.drug_embeddings
        self.g.nodes['disease'].data['feat'] = self.disease_embeddings
        self.g.nodes['gene'].data['feat'] = self.gene_embeddings

    def mask_node_features(self, mask_rate):
        drug_features = self.g.nodes['drug'].data['feat'].clone()
        disease_features = self.g.nodes['disease'].data['feat'].clone()
        
        num_mask_drugs = int(self.num_drugs * mask_rate)
        num_mask_diseases = int(self.num_diseases * mask_rate)
        
        drug_mask_idx = torch.randperm(self.num_drugs)[:num_mask_drugs].to(self.device)
        disease_mask_idx = torch.randperm(self.num_diseases)[:num_mask_diseases].to(self.device)
        
        if len(drug_mask_idx) > 0:
            drug_features[drug_mask_idx] = self.mask_token_drug.expand(num_mask_drugs, -1)
        
        if len(disease_mask_idx) > 0:
            disease_features[disease_mask_idx] = self.mask_token_disease.expand(num_mask_diseases, -1)

        self.g.nodes['drug'].data['mask_feat'] = drug_features
        self.g.nodes['disease'].data['mask_feat'] = disease_features
        
        masked_indices = {
            'drug': drug_mask_idx,
            'disease': disease_mask_idx
        }
        
        return masked_indices

    def multi_view_remask_decoding(self, encoder_output, masked_indices):
        original_drug_features = self.g.nodes['drug'].data['feat']
        original_disease_features = self.g.nodes['disease'].data['feat']

        total_loss = 0.0

        for view_idx in range(self.remask_view):
            remask_drug_embeds = encoder_output['drug'].clone()
            remask_disease_embeds = encoder_output['disease'].clone()

            num_remask_drugs = int(self.num_drugs * self.remask_rate)
            num_remask_diseases = int(self.num_diseases * self.remask_rate)

            remask_drug_idx = torch.randperm(self.num_drugs)[:num_remask_drugs].to(self.device)
            remask_disease_idx = torch.randperm(self.num_diseases)[:num_remask_diseases].to(self.device)
            
            if len(remask_drug_idx) > 0:
                remask_drug_embeds[remask_drug_idx] = self.dmask_token_drug.expand(num_remask_drugs, -1)
            
            if len(remask_disease_idx) > 0:
                remask_disease_embeds[remask_disease_idx] = self.dmask_token_disease.expand(num_remask_diseases, -1)
           
            decoder_input = {
                'drug': remask_drug_embeds,
                'disease': remask_disease_embeds
            }

            reconstructed = self.decoder(decoder_input)
            
            view_losses = []

            if len(masked_indices['drug']) > 0:
                drug_idx = masked_indices['drug']
                original = original_drug_features[drug_idx]
                reconstructed_drug = reconstructed['drug'][drug_idx]
                
                cos_sim = F.cosine_similarity(original, reconstructed_drug, dim=1)
                drug_loss = torch.pow(1 - cos_sim, self.scale_factor)
                view_losses.append(drug_loss)
            
            if len(masked_indices['disease']) > 0:
                disease_idx = masked_indices['disease']
                original = original_disease_features[disease_idx]
                reconstructed_disease = reconstructed['disease'][disease_idx]
                
                cos_sim = F.cosine_similarity(original, reconstructed_disease, dim=1)
                disease_loss = torch.pow(1 - cos_sim, self.scale_factor)
                view_losses.append(disease_loss)
            
            if len(view_losses) > 0:
                view_loss = torch.cat(view_losses).mean()
                total_loss += view_loss
        
        reconstruction_loss = total_loss / self.remask_view
        
        return reconstruction_loss

    def forward(self, drug_indices, disease_indices, current_epoch):
        mask_rate = self.mask_scheduler.get_mask_rate(current_epoch)

        masked_indices = self.mask_node_features(mask_rate)

        encoder_output = self.encoder(self.g, use_masked_features=True)

        drug_embeds = encoder_output['drug'][drug_indices]
        disease_embeds = encoder_output['disease'][disease_indices]

        prediction_scores = self.predictor(drug_embeds, disease_embeds)

        reconstruction_loss = self.multi_view_remask_decoding(encoder_output, masked_indices)

        return prediction_scores, reconstruction_loss

    def predict_scores(self, pairs):
        self.eval()
        with torch.no_grad():
            drug_indices = torch.LongTensor([p[0] for p in pairs]).to(self.g.device)
            disease_indices = torch.LongTensor([p[1] for p in pairs]).to(self.g.device)

            encoder_output = self.encoder(self.g, use_masked_features=False)

            drug_embeds = encoder_output['drug'][drug_indices]
            disease_embeds = encoder_output['disease'][disease_indices]
            
            scores = self.predictor(drug_embeds, disease_embeds)
            
        return scores.cpu().numpy()
    
    def clear_mask_features(self):
        for ntype in self.g.ntypes:
            if 'mask_feat' in self.g.nodes[ntype].data:
                del self.g.nodes[ntype].data['mask_feat']
    
    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)
        
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))