import os
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
from parse_args import args

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred_proba):
        y_pred_proba = y_pred_proba.cpu().numpy()
    
    y_true = y_true.astype(int)
    y_pred_proba = y_pred_proba.astype(float)
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics['auc'] = 0.0
    
    metrics['aupr'] = average_precision_score(y_true, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['confusion_matrix'] = {
        'TP': int(tp), 'TN': int(tn), 
        'FP': int(fp), 'FN': int(fn)
    }
    
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc, aupr, accuracy, precision, recall, f1, mcc = round(metrics['auc'], 4), round(metrics['aupr'] , 4), round(metrics['accuracy'] , 4), round(metrics['precision'] , 4), round(metrics['recall'] , 4), round(metrics['f1'] , 4), round(metrics['mcc'] , 4)
    
    return auc, aupr, accuracy, precision, recall, f1, mcc

class EarlyStopping:
    def __init__(self, min_delta=0):
        self.patience = args.patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
    
        return False

class Trainer:
    def __init__(self, model, device, log_folder_path):
        self.model = model
        self.device = device
        self.epochs = args.epochs
        self.log_folder_path = log_folder_path

        self.optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = args.epochs // 3, eta_min = args.lr / 10)

        self.criterion = torch.nn.BCELoss()

        self.early_stopping = EarlyStopping()
    
    def train_epoch(self, train_loader, current_epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            drug_ids = batch['drug_id'].to(self.device)
            disease_ids = batch['disease_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            scores, reconstruction_loss = self.model(drug_ids, disease_ids, current_epoch)
            bce_loss = self.criterion(scores, labels)

            loss = bce_loss + args.alpha * reconstruction_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()

            self.model.clear_mask_features()
            
        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                drug_ids = batch['drug_id'].to(self.device)
                disease_ids = batch['disease_id'].to(self.device)
                labels = batch['label'].to(self.device)

                pairs = [(drug_ids[i].item(), disease_ids[i].item()) for i in range(len(drug_ids))]
                scores = self.model.predict_scores(pairs)
                
                all_scores.extend(scores)
                all_labels.extend(labels.cpu().numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        auc, aupr, accuracy, precision, recall, f1, mcc = calculate_metrics(all_labels, all_scores, threshold=0.5)

        return (auc, aupr, accuracy, precision, recall, f1, mcc)

    def train(self, train_loader, val_loader, fold_i):
        best_metric = -float("inf")

        out_path = os.path.join(self.log_folder_path, "fold_{}".format(fold_i))

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        ckpt_path = os.path.join(out_path, "result.ckpt")
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            
            auc, aupr, accuracy, precision, recall, f1, mcc = self.evaluate(val_loader)
            
            self.scheduler.step()

            current_result = (epoch + 1, round(train_loss, 4), round(auc, 4), round(aupr, 4), round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4), round(mcc, 4))
            
            mix_factor = auc + aupr + mcc
            if mix_factor > best_metric:
                best_metric = mix_factor
                self.model.save(ckpt_path)

            if self.early_stopping(mix_factor):
                print('Early stopping triggered!')
                break
            
            if epoch % 10 == 0:
                print("\t".join(map(str, current_result)))
                print("\n")