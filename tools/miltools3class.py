import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from models import AttentionMIL3class

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

def undersample_bags(bags, labels, pids):
    """
    Undersample bags to balance 'infl', 'covid', and 'nc' classes.
    
    Args:
        bags: torch.Tensor of shape (n_bags, bag_size, feature_dim)
        labels: np.array of shape (n_bags,), values: 'infl', 'covid', 'nc'
        pids: np.array of shape (n_bags,)
    
    Returns:
        tuple: (sampled_bags, sampled_labels, sampled_pids)
    """
    label_map = {'infl': 0, 'covid': 1, 'nc': 2}
    labels_numeric = np.array([label_map[label] for label in labels])
    
    # Count bags for each class
    infl_indices = np.where(labels_numeric == 0)[0]
    covid_indices = np.where(labels_numeric == 1)[0]
    nc_indices = np.where(labels_numeric == 2)[0]
    n_infl = len(infl_indices)
    n_covid = len(covid_indices)
    n_nc = len(nc_indices)
    n_min = min(n_infl, n_covid, n_nc)
    
    if n_min == 0:
        raise ValueError("One of the classes has zero bags, cannot undersample.")
    
    # Randomly sample from each class
    np.random.seed(42)  # For reproducibility
    sampled_infl = np.random.choice(infl_indices, size=n_min, replace=False)
    sampled_covid = np.random.choice(covid_indices, size=n_min, replace=False)
    sampled_nc = np.random.choice(nc_indices, size=n_min, replace=False)
    sampled_indices = np.concatenate([sampled_infl, sampled_covid, sampled_nc])
    
    # Shuffle indices
    np.random.shuffle(sampled_indices)
    
    # Return sampled data
    return (bags[sampled_indices], 
            labels[sampled_indices], 
            pids[sampled_indices])

def specificity_score(y_true, y_pred, pos_label, labels):
    """
    Calculate specificity: TN / (TN + FP) for a given positive label.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Label to treat as positive
        labels: List of all possible labels
    
    Returns:
        float: Specificity score
    """
    y_true_bin = (y_true == pos_label).astype(int)
    y_pred_bin = (y_pred == pos_label).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def compute_metrics(labels, predictions, pids, labels_list=[0, 1, 2], label_names=['infl', 'covid', 'nc']):
    """
    Compute metrics for bags and pids for 3-class classification.
    
    Args:
        labels: np.array of true labels (0: infl, 1: covid, 2: nc)
        predictions: np.array of predicted labels (0: infl, 1: covid, 2: nc)
        pids: np.array of patient IDs
        labels_list: List of label indices
        label_names: List of label names
    
    Returns:
        dict: Bag-level and PID-level metrics
    """
    # Bag-level metrics
    bag_metrics = {}
    bag_metrics['confusion_matrix'] = confusion_matrix(labels, predictions, labels=labels_list).tolist()
    bag_metrics['accuracy'] = float(accuracy_score(labels, predictions))
    
    # Compute metrics for each class
    for metric_name in ['f1_score', 'precision', 'sensitivity', 'specificity']:
        scores = []
        for label, name in zip(labels_list, label_names):
            if metric_name == 'specificity':
                score = specificity_score(labels, predictions, pos_label=label, labels=labels_list)
            else:
                # Use average=None to get per-class scores, select score for pos_label
                scores_per_class = {
                    'f1_score': f1_score,
                    'precision': precision_score,
                    'sensitivity': recall_score
                }[metric_name](labels, predictions, labels=labels_list, average=None)
                score = float(scores_per_class[labels_list.index(label)])
            bag_metrics[f"{metric_name}_{name}"] = float(score)
            scores.append(score)
        bag_metrics[f"{metric_name}_mean"] = float(np.mean(scores))
    
    # PID-level majority voting
    pid_df = pd.DataFrame({
        'pid': pids,
        'pred': predictions,
        'true': labels,
        'pred_proba_infl': np.zeros(len(labels)),  # Placeholder, updated in evaluate_mil_model
        'pred_proba_covid': np.zeros(len(labels)),
        'pred_proba_nc': np.zeros(len(labels))
    })
    pid_agg = pid_df.groupby('pid').agg({
        'pred': lambda x: np.bincount(x, minlength=3).argmax(),
        'true': 'first',
        'pred_proba_infl': 'sum',
        'pred_proba_covid': 'sum',
        'pred_proba_nc': 'sum'
    }).reset_index()
    pid_predictions = pid_agg['pred'].values
    pid_true = pid_agg['true'].values
    
    # PID-level metrics
    pid_metrics = {}
    pid_metrics['confusion_matrix'] = confusion_matrix(pid_true, pid_predictions, labels=labels_list).tolist()
    pid_metrics['accuracy'] = float(accuracy_score(pid_true, pid_predictions))
    
    for metric_name in ['f1_score', 'precision', 'sensitivity', 'specificity']:
        scores = []
        for label, name in zip(labels_list, label_names):
            if metric_name == 'specificity':
                score = specificity_score(pid_true, pid_predictions, pos_label=label, labels=labels_list)
            else:
                scores_per_class = {
                    'f1_score': f1_score,
                    'precision': precision_score,
                    'sensitivity': recall_score
                }[metric_name](pid_true, pid_predictions, labels=labels_list, average=None)
                score = float(scores_per_class[labels_list.index(label)])
            pid_metrics[f"{metric_name}_{name}"] = float(score)
            scores.append(score)
        pid_metrics[f"{metric_name}_mean"] = float(np.mean(scores))
    
    return {
        'bag_metrics': bag_metrics,
        'pid_metrics': pid_metrics
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
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path

def plot_curves(train_losses, test_losses, test_accuracies, test_f1_scores, output_dir):
    """
    Plot and save loss and metric curves as PNG.
    """
    epochs = list(range(1, len(train_losses) + 1))
    
    # Loss curve
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
    
    # Accuracy and F1-Score curve
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

def save_label_counts(train_labels, test_labels, output_dir):
    """
    Save the count of each label in train and test data to JSON.
    """
    label_map = {'infl': 0, 'covid': 1, 'nc': 2}
    train_labels_numeric = np.array([label_map[label] for label in train_labels])
    test_labels_numeric = np.array([label_map[label] for label in test_labels])
    
    label_counts = {
        'train': {
            'infl': int(np.sum(train_labels_numeric == 0)),
            'covid': int(np.sum(train_labels_numeric == 1)),
            'nc': int(np.sum(train_labels_numeric == 2))
        },
        'test': {
            'infl': int(np.sum(test_labels_numeric == 0)),
            'covid': int(np.sum(test_labels_numeric == 1)),
            'nc': int(np.sum(test_labels_numeric == 2))
        }
    }
    
    counts_path = os.path.join(output_dir, 'label_counts.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(counts_path, 'w') as f:
        json.dump(label_counts, f, indent=4)
    return counts_path

def save_confusion_matrix(y_true, y_pred, output_file, labels=[0, 1, 2], label_names=['infl', 'covid', 'nc']):
    """
    Calculate and save confusion matrix as .npy and .json files.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_file: Path to save the confusion matrix (.npy extension)
        labels: List of label indices
        label_names: List of label names
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Save as .npy
    try:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        np.save(output_file, cm)
        print(f"Confusion matrix saved to {output_file}")
    except Exception as e:
        print(f"Error saving confusion matrix to {output_file}: {str(e)}")
    
    # Save as .json
    json_file = output_file.replace('.npy', '.json')
    try:
        cm_data = {
            'labels': label_names,
            'matrix': cm.tolist()
        }
        with open(json_file, 'w') as f:
            json.dump(cm_data, f, indent=4)
        print(f"Confusion matrix saved to {json_file}")
    except Exception as e:
        print(f"Error saving confusion matrix to {json_file}: {str(e)}")

def train_mil_model(train_bags, train_labels, train_pids, val_bags, val_labels, val_pids, feature_dim, epochs=50, lr=0.001, output_dir='results'):
    """
    Train the Attention MIL model for 3-class classification, saving the best model based on PID F1-Score.
    
    Args:
        train_bags: torch.Tensor of shape (n_train_bags, bag_size, feature_dim)
        train_labels: np.array of shape (n_train_bags,), values: 'infl', 'covid', 'nc'
        train_pids: np.array of shape (n_train_bags,)
        val_bags: torch.Tensor of shape (n_val_bags, bag_size, feature_dim)
        val_labels: np.array of shape (n_val_bags,), values: 'infl', 'covid', 'nc'
        val_pids: np.array of shape (n_val_bags,)
        feature_dim: int, dimension of input features
        epochs: int, number of training epochs
        lr: float, learning rate
        output_dir: str, directory to save outputs
    
    Returns:
        tuple: (model, best_metrics, best_epoch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL3class(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_bags, val_bags = train_bags.to(device), val_bags.to(device)
    label_map = {'infl': 0, 'covid': 1, 'nc': 2}
    train_labels_tensor = torch.tensor([label_map[label] for label in train_labels], dtype=torch.long).to(device)
    val_labels_tensor = torch.tensor([label_map[label] for label in val_labels], dtype=torch.long).to(device)
    
    best_pid_f1 = 0.0
    best_model_state = None
    best_metrics = None
    best_epoch = 0
    
    # Lists to store curves
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_f1_scores = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(train_bags)
        loss = criterion(logits, train_labels_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_bags)
            val_loss = criterion(val_logits, val_labels_tensor)
            val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
            val_predictions = np.argmax(val_probs, axis=1)
            test_losses.append(val_loss.item())
            
            # Compute test metrics
            val_metrics = compute_metrics(val_labels_tensor.cpu().numpy(), val_predictions, val_pids)
            test_accuracies.append(val_metrics['bag_metrics']['accuracy'])
            test_f1_scores.append(val_metrics['bag_metrics']['f1_score_mean'])
            
            # Update best model if PID F1-Score improves
            pid_f1 = val_metrics['pid_metrics']['f1_score_mean']
            if pid_f1 > best_pid_f1:
                best_pid_f1 = pid_f1
                best_model_state = model.state_dict()
                best_metrics = val_metrics
                best_epoch = epoch + 1
    
    # Save curves
    save_curve_data(train_losses, test_losses, test_accuracies, test_f1_scores, output_dir)
    plot_curves(train_losses, test_losses, test_accuracies, test_f1_scores, output_dir)
    
    model.load_state_dict(best_model_state)
    return model, best_metrics, best_epoch

def evaluate_mil_model(model, test_bags, test_labels, test_pids, output_dir='results'):
    """
    Evaluate the model for 3-class classification and compute metrics for bags and pids.
    
    Args:
        model: Trained AttentionMIL model
        test_bags: torch.Tensor of shape (n_test_bags, bag_size, feature_dim)
        test_labels: np.array of shape (n_test_bags,), values: 'infl', 'covid', 'nc'
        test_pids: np.array of shape (n_test_bags,)
        output_dir: str, directory to save outputs
    
    Returns:
        dict: Test metrics (bag_metrics, pid_metrics)
    """
    # Validate inputs
    if not all(label in ['infl', 'covid', 'nc'] for label in test_labels):
        print("Error: test_labels must contain only 'infl', 'covid', or 'nc'")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_bags = test_bags.to(device)
    
    # Convert labels to numeric
    label_map = {'infl': 0, 'covid': 1, 'nc': 2}
    test_labels_numeric = np.array([label_map[label] for label in test_labels])
    
    with torch.no_grad():
        logits, _ = model(test_bags)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probs, axis=1)
    
    # Compute metrics
    results = compute_metrics(test_labels_numeric, predictions, test_pids)
    
    # Update pid_metrics with probability sums
    pid_df = pd.DataFrame({
        'pid': test_pids,
        'pred': predictions,
        'true': test_labels_numeric,
        'pred_proba_infl': probs[:, 0],
        'pred_proba_covid': probs[:, 1],
        'pred_proba_nc': probs[:, 2]
    })
    pid_agg = pid_df.groupby('pid').agg({
        'pred': lambda x: np.bincount(x, minlength=3).argmax(),
        'true': 'first',
        'pred_proba_infl': 'sum',
        'pred_proba_covid': 'sum',
        'pred_proba_nc': 'sum'
    }).reset_index()
    pid_predictions = pid_agg['pred'].values
    pid_true = pid_agg['true'].values
    
    # Recalculate PID-level metrics with updated probabilities
    results['pid_metrics'] = compute_metrics(pid_true, pid_predictions, pid_agg['pid'].values)['pid_metrics']
    results['pid_metrics']['pred_proba_infl'] = pid_agg['pred_proba_infl'].values.tolist()
    results['pid_metrics']['pred_proba_covid'] = pid_agg['pred_proba_covid'].values.tolist()
    results['pid_metrics']['pred_proba_nc'] = pid_agg['pred_proba_nc'].values.tolist()
    
    # Save instance-level metrics
    os.makedirs(output_dir, exist_ok=True)
    instance_metrics_file = os.path.join(output_dir, 'metrics_summary.json')
    try:
        with open(instance_metrics_file, 'w') as f:
            json.dump(results['bag_metrics'], f, indent=4)
        print(f"Instance-level metrics saved to {instance_metrics_file}")
    except Exception as e:
        print(f"Error saving instance-level metrics: {str(e)}")
    
    # Save instance-level confusion matrix
    instance_cm_file = os.path.join(output_dir, 'confusion_matrix.npy')
    save_confusion_matrix(test_labels_numeric, predictions, instance_cm_file)
    
    # Save PID-level metrics
    pid_metrics_file = os.path.join(output_dir, 'pid_metrics_summary.json')
    try:
        with open(pid_metrics_file, 'w') as f:
            json.dump(results['pid_metrics'], f, indent=4)
        print(f"PID-level metrics saved to {pid_metrics_file}")
    except Exception as e:
        print(f"Error saving PID-level metrics: {str(e)}")
    
    # Save PID-level confusion matrix
    pid_cm_file = os.path.join(output_dir, 'pid_confusion_matrix.npy')
    save_confusion_matrix(pid_true, pid_predictions, pid_cm_file)
    
    return results

def evaluate_mil_3class(train_bags, train_labels, train_pids, test_bags, test_labels, test_pids, feature_dim, epochs=50, lr=0.001, output_dir='results'):
    """
    Main function to train and evaluate the MIL model for 3-class classification, with undersampling.
    
    Args:
        train_bags: torch.Tensor of shape (n_train_bags, bag_size, feature_dim) or list of np.ndarray
        train_labels: np.array of shape (n_train_bags,), values: 'infl', 'covid', 'nc'
        train_pids: np.array of shape (n_train_bags,)
        test_bags: torch.Tensor of shape (n_test_bags, bag_size, feature_dim) or list of np.ndarray
        test_labels: np.array of shape (n_test_bags,), values: 'infl', 'covid', 'nc'
        test_pids: np.array of shape (n_test_bags,)
        feature_dim: int, dimension of input features
        epochs: int, number of training epochs
        lr: float, learning rate
        output_dir: str, directory to save outputs
    
    Returns:
        tuple: (test_results, best_metrics, best_epoch)
    """
    # Validate inputs
    if not all(label in ['infl', 'covid', 'nc'] for label in train_labels) or \
       not all(label in ['infl', 'covid', 'nc'] for label in test_labels):
        print("Error: Labels must contain only 'infl', 'covid', or 'nc'")
        return
    
    # Convert bags to torch.Tensor if they are lists of numpy arrays
    if isinstance(train_bags, list):
        train_bags = np.array(train_bags)  # Convert list to numpy array
        train_bags = torch.tensor(train_bags, dtype=torch.float32)
    if isinstance(test_bags, list):
        test_bags = np.array(test_bags)  # Convert list to numpy array
        test_bags = torch.tensor(test_bags, dtype=torch.float32)
    
    # Undersample training data
    try:
        train_bags, train_labels, train_pids = undersample_bags(train_bags, train_labels, train_pids)
    except ValueError as e:
        print(f"Error in undersampling: {str(e)}")
        return
    
    # Save label counts (after undersampling for train)
    save_label_counts(train_labels, test_labels, output_dir)
    
    # Train the model
    model, best_metrics, best_epoch = train_mil_model(
        train_bags, train_labels, train_pids, 
        test_bags, test_labels, test_pids, 
        feature_dim, epochs, lr, output_dir
    )
    
    # Evaluate the model on test data
    test_results = evaluate_mil_model(model, test_bags, test_labels, test_pids, output_dir)
    
    # Save results to JSON
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
    
    # Print best validation metrics
    print(f"\nBest Validation PID F1-Score Metrics (Epoch {best_epoch}):")
    print("Bag-level Metrics:")
    print(f"Confusion Matrix:\n{np.array(best_metrics['bag_metrics']['confusion_matrix'])}")
    print(f"Accuracy: {best_metrics['bag_metrics']['accuracy']:.4f}")
    for name in ['infl', 'covid', 'nc']:
        print(f"F1-Score ({name}): {best_metrics['bag_metrics'][f'f1_score_{name}']:.4f}")
        print(f"Precision ({name}): {best_metrics['bag_metrics'][f'precision_{name}']:.4f}")
        print(f"Sensitivity ({name}): {best_metrics['bag_metrics'][f'sensitivity_{name}']:.4f}")
        print(f"Specificity ({name}): {best_metrics['bag_metrics'][f'specificity_{name}']:.4f}")
    print(f"F1-Score Mean: {best_metrics['bag_metrics']['f1_score_mean']:.4f}")
    print(f"Precision Mean: {best_metrics['bag_metrics']['precision_mean']:.4f}")
    print(f"Sensitivity Mean: {best_metrics['bag_metrics']['sensitivity_mean']:.4f}")
    print(f"Specificity Mean: {best_metrics['bag_metrics']['specificity_mean']:.4f}")
    
    print("PID-level Metrics:")
    print(f"Confusion Matrix:\n{np.array(best_metrics['pid_metrics']['confusion_matrix'])}")
    print(f"Accuracy: {best_metrics['pid_metrics']['accuracy']:.4f}")
    for name in ['infl', 'covid', 'nc']:
        print(f"F1-Score ({name}): {best_metrics['pid_metrics'][f'f1_score_{name}']:.4f}")
        print(f"Precision ({name}): {best_metrics['pid_metrics'][f'precision_{name}']:.4f}")
        print(f"Sensitivity ({name}): {best_metrics['pid_metrics'][f'sensitivity_{name}']:.4f}")
        print(f"Specificity ({name}): {best_metrics['pid_metrics'][f'specificity_{name}']:.4f}")
    print(f"F1-Score Mean: {best_metrics['pid_metrics']['f1_score_mean']:.4f}")
    print(f"Precision Mean: {best_metrics['pid_metrics']['precision_mean']:.4f}")
    print(f"Sensitivity Mean: {best_metrics['pid_metrics']['sensitivity_mean']:.4f}")
    print(f"Specificity Mean: {best_metrics['pid_metrics']['specificity_mean']:.4f}")
    
    # Print test metrics
    print("\nTest Metrics:")
    print("Bag-level Metrics:")
    print(f"Confusion Matrix:\n{np.array(test_results['bag_metrics']['confusion_matrix'])}")
    print(f"Accuracy: {test_results['bag_metrics']['accuracy']:.4f}")
    for name in ['infl', 'covid', 'nc']:
        print(f"F1-Score ({name}): {test_results['bag_metrics'][f'f1_score_{name}']:.4f}")
        print(f"Precision ({name}): {test_results['bag_metrics'][f'precision_{name}']:.4f}")
        print(f"Sensitivity ({name}): {test_results['bag_metrics'][f'sensitivity_{name}']:.4f}")
        print(f"Specificity ({name}): {test_results['bag_metrics'][f'specificity_{name}']:.4f}")
    print(f"F1-Score Mean: {test_results['bag_metrics']['f1_score_mean']:.4f}")
    print(f"Precision Mean: {test_results['bag_metrics']['precision_mean']:.4f}")
    print(f"Sensitivity Mean: {test_results['bag_metrics']['sensitivity_mean']:.4f}")
    print(f"Specificity Mean: {test_results['bag_metrics']['specificity_mean']:.4f}")
    
    print("PID-level Metrics:")
    print(f"Confusion Matrix:\n{np.array(test_results['pid_metrics']['confusion_matrix'])}")
    print(f"Accuracy: {test_results['pid_metrics']['accuracy']:.4f}")
    for name in ['infl', 'covid', 'nc']:
        print(f"F1-Score ({name}): {test_results['pid_metrics'][f'f1_score_{name}']:.4f}")
        print(f"Precision ({name}): {test_results['pid_metrics'][f'precision_{name}']:.4f}")
        print(f"Sensitivity ({name}): {test_results['pid_metrics'][f'sensitivity_{name}']:.4f}")
        print(f"Specificity ({name}): {test_results['pid_metrics'][f'specificity_{name}']:.4f}")
    print(f"F1-Score Mean: {test_results['pid_metrics']['f1_score_mean']:.4f}")
    print(f"Precision Mean: {test_results['pid_metrics']['precision_mean']:.4f}")
    print(f"Sensitivity Mean: {test_results['pid_metrics']['sensitivity_mean']:.4f}")
    print(f"Specificity Mean: {test_results['pid_metrics']['specificity_mean']:.4f}")
    
    return test_results, best_metrics, best_epoch
