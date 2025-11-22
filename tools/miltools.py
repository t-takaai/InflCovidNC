import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import json
import os
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_recall_curve, roc_curve, auc

from models import AttentionMIL

def split_data_by_pid(df_X, df_y, chunk_size=100):
    """
    Split df_X into chunks of specified size based on unique pids in df_y.
    Return chunks of df_X as a list of NumPy arrays, and corresponding labels and pids
    as single NumPy arrays with one value per chunk (size equals number of chunks).
    Chunks with fewer than chunk_size samples are discarded.
    
    Parameters:
    df_X (pd.DataFrame): Input features dataframe
    df_y (pd.DataFrame): Dataframe with 'pid', 'label', 'group' columns
    chunk_size (int): Number of samples per chunk
    
    Returns:
    tuple: (list of np.ndarray for X chunks, np.ndarray for labels, np.ndarray for pids)
    """
    # Ensure df_X and df_y have the same number of rows
    assert len(df_X) == len(df_y), "df_X and df_y must have the same number of rows"
    
    # Reset indices to ensure alignment
    df_X = df_X.reset_index(drop=True)
    df_y = df_y.reset_index(drop=True)
    
    # Group by 'pid' and get indices for each pid
    pid_groups = df_y.groupby('pid').indices
    
    X_chunks = []
    label_values = []
    pid_values = []
    
    # Process each pid
    for pid in pid_groups:
        indices = pid_groups[pid]
        # For each chunk of indices
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i:i + chunk_size]
            # Only include chunks with exactly chunk_size samples
            if len(chunk_indices) == chunk_size:
                # Extract chunk from df_X
                X_chunk = df_X.iloc[chunk_indices].to_numpy()
                # Extract the first label for the chunk (assuming labels are consistent within pid)
                label_value = df_y.iloc[chunk_indices[0]]['label']
                # Extract the pid for the chunk
                pid_value = df_y.iloc[chunk_indices[0]]['pid']
                # Append to output lists
                X_chunks.append(X_chunk)
                label_values.append(label_value)
                pid_values.append(pid_value)
    
    # Convert label and pid lists to NumPy arrays
    labels_array = np.array(label_values)
    pids_array = np.array(pid_values)
    
    return X_chunks, labels_array, pids_array

def undersample_bags(bags, labels, pids, label_map={'covid': 1, 'nc': 0}):
    """
    Undersample bags to balance 'covid' and 'nc' classes while preserving at least one bag per PID.
    """
    labels_binary = np.array([label_map[label] for label in labels])
    
    # 初期のPID数とバッグ数をログ
    print(f"アンダーサンプリング前: {len(np.unique(pids))} PID, {len(bags)} バッグ")
    print(f"初期クラス分布: covid={np.sum(labels_binary == 1)}, nc={np.sum(labels_binary == 0)}")
    
    pid_df = pd.DataFrame({'pid': pids, 'index': range(len(pids)), 'label': labels_binary})
    
    # PIDごとに少なくとも1バッグを選択
    selected_indices = []
    for pid in np.unique(pids):
        pid_indices = pid_df[pid_df['pid'] == pid]['index'].values
        if len(pid_indices) > 0:
            np.random.seed(42)
            selected_indices.append(np.random.choice(pid_indices))
    
    # クラスごとのバッグ数をカウント
    infl_indices = np.where(labels_binary == 1)[0]
    nc_indices = np.where(labels_binary == 0)[0]
    n_infl = len(infl_indices)
    n_nc = len(nc_indices)
    n_min = min(n_infl, n_nc)
    
    if n_min == 0:
        print("警告: 一方のクラスにバッグがありません、PID保持バッグのみで進行します。")
    else:
        # 追加のバッグを均衡させる
        infl_pids = pid_df[pid_df['label'] == 1]['pid'].unique()
        nc_pids = pid_df[pid_df['label'] == 0]['pid'].unique()
        additional_infl_needed = max(0, n_min - len(infl_pids))
        additional_nc_needed = max(0, n_min - len(nc_pids))
        
        np.random.seed(42)
        available_infl = [i for i in infl_indices if i not in selected_indices]
        available_nc = [i for i in nc_indices if i not in selected_indices]
        additional_infl = np.random.choice(available_infl, size=min(additional_infl_needed, len(available_infl)), replace=False) if available_infl else []
        additional_nc = np.random.choice(available_nc, size=min(additional_nc_needed, len(available_nc)), replace=False) if available_nc else []
        selected_indices.extend(additional_infl)
        selected_indices.extend(additional_nc)
    
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    
    # 最終的なPID数とバッグ数をログ
    sampled_bags = bags[selected_indices]
    sampled_labels = labels[selected_indices]
    sampled_pids = pids[selected_indices]
    print(f"アンダーサンプリング後: {len(np.unique(sampled_pids))} PID, {len(sampled_bags)} バッグ")
    print(f"最終クラス分布: covid={np.sum([label_map[l] for l in sampled_labels] == 1)}, nc={np.sum([label_map[l] for l in sampled_labels] == 0)}")
    
    return sampled_bags, sampled_labels, sampled_pids

def compute_metrics(labels, predictions, pids, probs):
    """
    Compute metrics for bags and pids, including PID-level probabilities.
    Returns metrics and PID-level true/predicted labels and probabilities.
    """
    if len(labels) == 0 or len(predictions) == 0:
        print("警告: ラベルまたは予測が空です、デフォルトメトリクスを返します。")
        return {
            'bag_metrics': {
                'confusion_matrix': [[0, 0], [0, 0]],
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0
            },
            'pid_metrics': {
                'confusion_matrix': [[0, 0], [0, 0]],
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0
            },
            'pid_labels': {
                'pid_true': [],
                'pid_predictions': [],
                'pid_probs': [],
                'pids': []
            }
        }
    
    # Bag-level metrics
    bag_conf_matrix = confusion_matrix(labels, predictions)
    bag_accuracy = accuracy_score(labels, predictions)
    bag_f1 = f1_score(labels, predictions, zero_division=0)
    bag_precision = precision_score(labels, predictions, zero_division=0)
    bag_sensitivity = recall_score(labels, predictions, zero_division=0)
    bag_specificity = recall_score(labels, predictions, pos_label=0, zero_division=0) if np.sum(labels == 0) > 0 else 0
    
    # PID-level majority voting and probability aggregation
    pid_df = pd.DataFrame({'pid': pids, 'pred': predictions, 'true': labels, 'prob': probs})
    pid_agg = pid_df.groupby('pid').agg({
        'pred': lambda x: np.bincount(x).argmax(),
        'true': 'first',
        'prob': 'mean'  # PIDごとの確率を平均で集約
    }).reset_index()
    pid_predictions = pid_agg['pred'].values
    pid_true = pid_agg['true'].values
    pid_probs = pid_agg['prob'].values
    
    # PID-level metrics
    pid_conf_matrix = confusion_matrix(pid_true, pid_predictions, labels=[0, 1])
    pid_accuracy = accuracy_score(pid_true, pid_predictions)
    pid_f1 = f1_score(pid_true, pid_predictions, zero_division=0)
    pid_precision = precision_score(pid_true, pid_predictions, zero_division=0)
    pid_sensitivity = recall_score(pid_true, pid_predictions, zero_division=0)
    pid_specificity = recall_score(pid_true, pid_predictions, pos_label=0, zero_division=0) if np.sum(pid_true == 0) > 0 else 0
    
    return {
        'bag_metrics': {
            'confusion_matrix': bag_conf_matrix.tolist(),
            'accuracy': float(bag_accuracy),
            'f1_score': float(bag_f1),
            'precision': float(bag_precision),
            'sensitivity': float(bag_sensitivity),
            'specificity': float(bag_specificity)
        },
        'pid_metrics': {
            'confusion_matrix': pid_conf_matrix.tolist(),
            'accuracy': float(pid_accuracy),
            'f1_score': float(pid_f1),
            'precision': float(pid_precision),
            'sensitivity': float(pid_sensitivity),
            'specificity': float(pid_specificity)
        },
        'pid_labels': {
            'pid_true': pid_true.tolist(),
            'pid_predictions': pid_predictions.tolist(),
            'pid_probs': pid_probs.tolist(),
            'pids': pid_agg['pid'].tolist()
        }
    }

def save_curve_data(train_losses, test_losses, test_accuracies, test_f1_scores, output_dir):
    """
    Save loss and metric curves to CSV.
    """
    epochs = list(range(1, len(train_losses) + 1))
    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'test_loss': test_losses,
        'test_accuracy': test_accuracies,
        'test_f1_score': test_f1_scores
    })
    csv_path = os.path.join(output_dir, 'curves.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

def plot_curves(train_losses, test_losses, test_accuracies, test_f1_scores, output_dir):
    """
    Plot and save loss and metric curves as PNG.
    """
    epochs = list(range(1, len(train_losses) + 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.plot(epochs, test_f1_scores, label='Test F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Test Accuracy and F1-Score Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_curve.png'))
    plt.close()

def save_label_counts(train_labels, test_labels, output_dir, label_map={'covid': 1, 'nc': 0}):
    """
    Save the count of each label in train and test data to JSON.
    """
    train_labels_binary = np.array([label_map[label] for label in train_labels])
    test_labels_binary = np.array([label_map[label] for label in test_labels])
    
    label_counts = {
        'train': {
            'covid': int(np.sum(train_labels_binary == 1)),
            'nc': int(np.sum(train_labels_binary == 0))
        },
        'test': {
            'covid': int(np.sum(test_labels_binary == 1)),
            'nc': int(np.sum(test_labels_binary == 0))
        }
    }
    
    counts_path = os.path.join(output_dir, 'label_counts.json')
    with open(counts_path, 'w') as f:
        json.dump(label_counts, f, indent=4)
    return counts_path

def train_mil_model(train_bags, train_labels, train_pids, val_bags, val_labels, val_pids, feature_dim, epochs=50, lr=0.001, output_dir='results'):
    """
    Train the Attention MIL model, saving the best model based on PID F1-Score across all epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    train_bags, val_bags = train_bags.to(device), val_bags.to(device)
    train_labels, val_labels = train_labels.to(device), val_labels.to(device)
    
    best_pid_f1 = 0.0
    best_model_state = None
    best_metrics = None
    best_epoch = 0
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_f1_scores = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(train_bags)
        loss = criterion(logits.squeeze(), train_labels.float())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_bags)
            val_loss = criterion(val_logits.squeeze(), val_labels.float())
            val_probs = torch.sigmoid(val_logits).squeeze().cpu().numpy()
            val_predictions = (val_probs >= 0.5).astype(int)
            test_losses.append(val_loss.item())
            
            val_metrics = compute_metrics(val_labels.cpu().numpy(), val_predictions, val_pids, val_probs)
            test_accuracies.append(val_metrics['bag_metrics']['accuracy'])
            test_f1_scores.append(val_metrics['bag_metrics']['f1_score'])
            
            pid_f1 = val_metrics['pid_metrics']['f1_score']
            if pid_f1 > best_pid_f1:
                best_pid_f1 = pid_f1
                best_model_state = model.state_dict()
                best_metrics = val_metrics
                best_epoch = epoch + 1
    
    os.makedirs(output_dir, exist_ok=True)
    save_curve_data(train_losses, test_losses, test_accuracies, test_f1_scores, output_dir)
    plot_curves(train_losses, test_losses, test_accuracies, test_f1_scores, output_dir)
    
    model.load_state_dict(best_model_state)
    return model, best_metrics, best_epoch

def evaluate_mil_model(model, test_bags, test_labels, test_pids, label_map={'covid': 1, 'nc': 0}):
    """
    Evaluate the model and compute metrics for bags and pids, including PID-level probabilities.
    Returns metrics including pid_true, pid_predictions, and pid_probs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_bags = test_bags.to(device)
    
    with torch.no_grad():
        logits, _ = model(test_bags)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        predictions = (probs >= 0.5).astype(int)
    
    test_labels_binary = np.array([label_map[label] for label in test_labels])
    
    results = compute_metrics(test_labels_binary, predictions, test_pids, probs)
    
    return results

def evaluate_MIL(train_bags, train_labels, train_pids, test_bags, test_labels, test_pids, feature_dim, epochs=50, lr=0.001, output_dir='results', label_map={'covid': 1, 'nc': 0}):
    """
    Main function to train and evaluate the MIL model, with undersampling of training bags.
    Saves PID true, predicted labels, and probabilities to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_bags, train_labels, train_pids = undersample_bags(train_bags, train_labels, train_pids, label_map)
    
    save_label_counts(train_labels, test_labels, output_dir, label_map)
    
    train_labels_tensor = torch.tensor([label_map[label] for label in train_labels], dtype=torch.long)
    test_labels_tensor = torch.tensor([label_map[label] for label in test_labels], dtype=torch.long)
    
    model, best_metrics, best_epoch = train_mil_model(
        train_bags, train_labels_tensor, train_pids, 
        test_bags, test_labels_tensor, test_pids, 
        feature_dim, epochs, lr, output_dir
    )
    
    test_results = evaluate_mil_model(model, test_bags, test_labels, test_pids, label_map)
    
    results_dict = {
        'best_validation_metrics': {
            'epoch': best_epoch,
            'metrics': best_metrics
        },
        'test_metrics': test_results
    }
    results_path = os.path.join(output_dir, 'mil_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    pid_labels = test_results['pid_labels']
    pid_output_dict = {
        'pids': pid_labels['pids'],
        'pid_true': pid_labels['pid_true'],
        'pid_predictions': pid_labels['pid_predictions'],
        'pid_probs': pid_labels['pid_probs']
    }
    pid_output_path = os.path.join(output_dir, 'pid_predictions.json')
    with open(pid_output_path, 'w') as f:
        json.dump(pid_output_dict, f, indent=4)
    
    print(f"\nBest Validation PID F1-Score Metrics (Epoch {best_epoch}):")
    print("Bag-level Metrics:")
    print(f"Confusion Matrix:\n{np.array(best_metrics['bag_metrics']['confusion_matrix'])}")
    print(f"Accuracy: {best_metrics['bag_metrics']['accuracy']:.4f}")
    print(f"F1-Score: {best_metrics['bag_metrics']['f1_score']:.4f}")
    print(f"Precision: {best_metrics['bag_metrics']['precision']:.4f}")
    print(f"Sensitivity: {best_metrics['bag_metrics']['sensitivity']:.4f}")
    print(f"Specificity: {best_metrics['bag_metrics']['specificity']:.4f}")
    print("PID-level Metrics:")
    print(f"Confusion Matrix:\n{np.array(best_metrics['pid_metrics']['confusion_matrix'])}")
    print(f"Accuracy: {best_metrics['pid_metrics']['accuracy']:.4f}")
    print(f"F1-Score: {best_metrics['pid_metrics']['f1_score']:.4f}")
    print(f"Precision: {best_metrics['pid_metrics']['precision']:.4f}")
    print(f"Sensitivity: {best_metrics['pid_metrics']['sensitivity']:.4f}")
    print(f"Specificity: {best_metrics['pid_metrics']['specificity']:.4f}")
    
    print("\nTest Metrics:")
    print("Bag-level Metrics:")
    print(f"Confusion Matrix:\n{np.array(test_results['bag_metrics']['confusion_matrix'])}")
    print(f"Accuracy: {test_results['bag_metrics']['accuracy']:.4f}")
    print(f"F1-Score: {test_results['bag_metrics']['f1_score']:.4f}")
    print(f"Precision: {test_results['bag_metrics']['precision']:.4f}")
    print(f"Sensitivity: {test_results['bag_metrics']['sensitivity']:.4f}")
    print(f"Specificity: {test_results['bag_metrics']['specificity']:.4f}")
    print("PID-level Metrics:")
    print(f"Confusion Matrix:\n{np.array(test_results['pid_metrics']['confusion_matrix'])}")
    print(f"Accuracy: {test_results['pid_metrics']['accuracy']:.4f}")
    print(f"F1-Score: {test_results['pid_metrics']['f1_score']:.4f}")
    print(f"Precision: {test_results['pid_metrics']['precision']:.4f}")
    print(f"Sensitivity: {test_results['pid_metrics']['sensitivity']:.4f}")
    print(f"Specificity: {test_results['pid_metrics']['specificity']:.4f}")
    
    print(f"\nPID true, predicted labels, and probabilities saved to {pid_output_path}")
    
    return test_results, best_metrics, best_epoch
