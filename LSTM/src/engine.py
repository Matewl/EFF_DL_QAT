import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from IPython.display import clear_output

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(x, lengths)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device).float().unsqueeze(1)
            
            outputs = model(x, lengths)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    try:
        roc_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        roc_auc = 0.5
        
    return avg_loss, roc_auc

def train_model(model, train_loader, val_loader, epochs, lr, device, 
                method_name="Training", all_histories=None, weight_decay=0.0,
                quant_lr_multiplier=1.0):
    
    # Разделяем параметры
    base_params, quant_params = [], []
    for name, param in model.named_parameters():
        if 'quant' in name or 'alpha' in name or name.endswith('.s') or name.endswith('.scale') or 'V' in name:
            quant_params.append(param)
        else:
            base_params.append(param)
            
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': lr},
        {'params': quant_params, 'lr': lr * quant_lr_multiplier}
    ], weight_decay=weight_decay)
    
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    if all_histories is not None:
        all_histories[method_name] = history
    else:
        all_histories = {method_name: history}
        
    colors = plt.cm.tab10.colors
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.set_title(f'Loss (Эпоха {epoch+1}/{epochs})', fontsize=14)
        ax1.set_xlabel('Эпоха', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2.set_title(f'ROC-AUC (Эпоха {epoch+1}/{epochs})', fontsize=14)
        ax2.set_xlabel('Эпоха', fontsize=12)
        ax2.set_ylabel('ROC-AUC', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        for i, (name, hist) in enumerate(all_histories.items()):
            if len(hist['val_loss']) == 0:
                continue
                
            c = colors[i % len(colors)]
            lw = 3.0 if name == method_name else 1.5
            alpha = 1.0 if name == method_name else 0.5
            
            epochs_range = range(1, len(hist['val_loss']) + 1)
            
            if len(hist['train_loss']) > 0:
                ax1.plot(epochs_range, hist['train_loss'], color=c, linestyle='--', 
                         alpha=alpha, linewidth=lw*0.8, label=f'{name} (Train)')
            
            ax1.plot(epochs_range, hist['val_loss'], color=c, marker='s', 
                     alpha=alpha, linewidth=lw, label=f'{name} (Val)')
            
            if len(hist['val_auc']) > 0:
                ax2.plot(epochs_range, hist['val_auc'], color=c, marker='o', 
                         alpha=alpha, linewidth=lw, label=name)
        
        handles1, labels1 = ax1.get_legend_handles_labels()
        by_label1 = dict(zip(labels1, handles1))
        ax1.legend(by_label1.values(), by_label1.keys(), loc='upper right', fontsize=9)
        
        handles2, labels2 = ax2.get_legend_handles_labels()
        by_label2 = dict(zip(labels2, handles2))
        ax2.legend(by_label2.values(), by_label2.keys(), loc='lower right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
    return history