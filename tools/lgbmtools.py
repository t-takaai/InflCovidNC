import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import os
import sys
import seaborn as sns
import random
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, BaseCrossValidator, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb


def save_clrep_confmat(y_test, y_pred, labels, filenamehead, outputdir):
    """
    Take y_test and y_pred, compute classification report and confusion matrix,
    and save them as CSV files.
    """
    clrep = classification_report(y_test, y_pred, output_dict=True)
    clrepname = filenamehead + '_clrep.csv'
    clreppath = outputdir / clrepname
    df_clrep = pd.DataFrame(clrep)
    df_clrep.to_csv(clreppath)

    confmat = confusion_matrix(y_test, y_pred)
    confmatname = filenamehead + '_confmat.csv'
    confmatpath = outputdir / confmatname
    df_confmat = pd.DataFrame(confmat, index=labels, columns=labels)
    df_confmat.to_csv(confmatpath)
    
    return clrep, confmat


def pred_bypid(
    pids_test,  # PIDs in the test data
    z_test,     # PID for each instance in the test data
    y_pred      # Predictions for each instance in the test data
):
    """Predict per PID by majority vote."""
    pidlabels_pred = []
    for pid in pids_test:  # Iterate over each PID in the test data
        y_pred_bypid = y_pred[z_test == pid]  # Predictions for this PID
        count = Counter(y_pred_bypid)
        label_pred = count.most_common()[0][0]  # Most frequent predicted label for this PID
        pidlabels_pred.append(label_pred)       # Use the most frequent label as the PID-level prediction
        
    return pidlabels_pred


def pred_prob_bypid(
    pids_test,  # PIDs in the test data
    z_test,     # PID for each instance in the test data
    y_pred,     # Predictions for each instance in the test data
    labels      # Label names
):
    """Compute prediction probabilities per PID (ratio of predicted labels)."""
    pidlabels_pred_prob = []
    for pid in pids_test:  # Iterate over each PID in the test data
        y_pred_bypid = y_pred[z_test == pid]  # Predictions for this PID
        # Counter is a dict subclass where keys are elements and values are their occurrence counts
        count = Counter(y_pred_bypid)
        pidlabels_pred_prob.append(count)
    df_pidlabels_pred_prob = pd.DataFrame(pidlabels_pred_prob, columns=labels)
    # Convert to row-wise ratios
    df_pidlabels_pred_prob = df_pidlabels_pred_prob.apply(lambda x: x / x.sum(), axis=1)
    
    return df_pidlabels_pred_prob


def pred_ratio_bypid(
    pids_test,  # PIDs in the test data
    z_test,     # PID for each instance in the test data
    y_pred,     # Predictions for each instance in the test data
    labels      # Label names
):
    """Compute the ratio of each label per PID."""

    pid_labels_ratio = []
    for pid in pids_test:  # Iterate over each PID in the test data
        y_pred_bypid = y_pred[z_test == pid]  # Predictions for this PID
        count = Counter(y_pred_bypid)
        # Total count
        total_count = sum(count.values())
        # Compute ratios
        ratios = [count[label] / total_count for label in labels]
        pid_labels_ratio.append(ratios)

    return np.array(pid_labels_ratio)


def train_pred_prob_lgbm(X_train, y_train, X_valid, y_valid, X_test):
    """
    Train LightGBM with X_train, y_train and validation set (X_valid, y_valid),
    then predict probabilities on X_test.
    """
    lgb_params = {
        'n_estimators': 10000,  # Use a large number and rely on early stopping
        'random_state': 0,
        'verbose': -1
    }
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=1000, verbose=False)]
    )
    y_pred_prob = clf.predict_proba(X_test)  # Predicted probabilities
    
    return y_pred_prob


def sensitivity(cm):
    """
    Compute sensitivity (recall, TPR) from a confusion matrix.
    """
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    Sensitivity = TP / (TP + FN)
    
    return np.concatenate([Sensitivity, np.mean(Sensitivity).reshape(-1)])


def specificity(cm):
    """
    Compute specificity (TNR) from a confusion matrix.
    """
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    Specificity = TN / (TN + FP)
    
    return np.concatenate([Specificity, np.mean(Specificity).reshape(-1)])


def sensitivity_specificity(cm, labels):
    """
    Compute sensitivity and specificity from a confusion matrix,
    and return them together with their macro averages as a DataFrame.
    """
    df = pd.DataFrame(
        [sensitivity(cm), specificity(cm)],
        index=['sensitivity', 'specificity'],
        columns=tuple(labels) + ('macro avg',)
    )
    return df


def calc_save_ave(experiment_dir):
    """
    Compute and save sensitivity/specificity and
    macro averages of f1-score, precision, recall, sensitivity, specificity
    over all experiments under experiment_dir.
    """

    # Compute and save sensitivity and specificity
    directories = sorted(experiment_dir.glob('ex-*'))
    
    for directory in directories:
        for base_name in ('case_nonr', 'pid_nonr'):
            confmat_name = base_name + '_confmat.csv'
            confmat_path = directory / confmat_name
            try:
                df_confmat = pd.read_csv(confmat_path, index_col=0)
                confmat = df_confmat.values
                df_sensitivity_specificity = sensitivity_specificity(confmat, labels)
                sensitivity_specificity_name = base_name + '_sensitivity_specificity.csv'
                sensitivity_specificity_path = directory / sensitivity_specificity_name
                df_sensitivity_specificity.to_csv(sensitivity_specificity_path)
            except FileNotFoundError as e:
                print(e)
                
    # Compute and save means of f1-score, precision, recall, sensitivity, specificity
    total_path = experiment_dir / 'total'  # Directory where average metrics will be saved
    os.makedirs(total_path, exist_ok=True)
    ex_dirs = sorted(experiment_dir.glob('ex-*'))
    
    for base_name in ('case_nonr', 'pid_nonr'):
        df_scores = []
        score = 'accuracy'
        values = {}
        for ex_dir in ex_dirs:
            clrep_name = base_name + '_clrep.csv'
            clrep_path = ex_dir / clrep_name
            try:
                df_clrep = pd.read_csv(clrep_path, index_col=0)
                value = df_clrep.loc['precision'][score]
                values[str(ex_dir.name)] = value
            except FileNotFoundError as e:
                values[str(ex_dir.name)] = np.nan
                print(e)
        df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
    
        for score in ('f1-score', 'precision', 'recall'):
            values = {}
            for ex_dir in ex_dirs:
                clrep_name = base_name + '_clrep.csv'
                clrep_path = ex_dir / clrep_name
                try:
                    df_clrep = pd.read_csv(clrep_path, index_col=0)
                    value = df_clrep.loc[score]['macro avg']
                    values[str(ex_dir.name)] = value
                except FileNotFoundError as e:
                    values[str(ex_dir.name)] = np.nan
                    print(e)
            df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
            
        for score in ('sensitivity', 'specificity'):
            values = {}
            for ex_dir in ex_dirs:
                clrep_name = base_name + '_sensitivity_specificity.csv'
                clrep_path = ex_dir / clrep_name
                try:
                    df_clrep = pd.read_csv(clrep_path, index_col=0)
                    value = df_clrep.loc[score]['macro avg']
                    values[str(ex_dir.name)] = value
                except FileNotFoundError as e:
                    values[str(ex_dir.name)] = np.nan
                    print(e)
            df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
    
        # Aggregate metrics
        df_scores = pd.concat(df_scores, axis=1)
        print(base_name)
        scores_name = base_name + '_meanscores.csv'
        scores_path = total_path / scores_name
        df_scores.to_csv(scores_path)

        # Histograms
        fig = plt.figure(figsize=(25, 25))
        for i, col in enumerate(df_scores.columns):
            ax = plt.subplot2grid((5, 3), (i // 3, i % 3))
            sns.histplot(df_scores[col], ax=ax)
            ax.set_title(col)
        plt.show()
        
        # Descriptive statistics and max values
        df_describe = df_scores.describe()
        files = []
        for score in ('accuracy', 'f1-score', 'precision', 'recall', 'sensitivity', 'specificity'):
            filename = df_scores.loc[[df_scores[score].idxmax()]].index[0]
            files.append(filename)
        df_describe.loc['max file'] = files
        display(df_describe)
        describe_name = base_name + '_describe.csv'
        describe_path = total_path / describe_name
        df_describe.to_csv(describe_path) 
        
        max_recall_ex = df_scores.loc[[df_scores['recall'].idxmax()]].index[0]
        confmat_name = base_name + '_confmat.csv'
        confmat_path = experiment_dir / max_recall_ex / confmat_name
        df_confmat = pd.read_csv(confmat_path, index_col=0)
        display(df_confmat)


def calc_save_ave_2class(experiment_dir, labels, target_label='macro avg'):
    """
    Same as calc_save_ave, but explicitly for binary (2-class) classification,
    and allows specifying which column (target_label) to use
    from the classification report (e.g., 'macro avg').
    """

    # Compute and save sensitivity and specificity
    directories = sorted(experiment_dir.glob('ex-*'))
    
    for directory in directories:
        for base_name in ('case_nonr', 'pid_nonr'):
            confmat_name = base_name + '_confmat.csv'
            confmat_path = directory / confmat_name
            try:
                df_confmat = pd.read_csv(confmat_path, index_col=0)
                confmat = df_confmat.values
                df_sensitivity_specificity = sensitivity_specificity(confmat, labels)
                sensitivity_specificity_name = base_name + '_sensitivity_specificity.csv'
                sensitivity_specificity_path = directory / sensitivity_specificity_name
                df_sensitivity_specificity.to_csv(sensitivity_specificity_path)
            except FileNotFoundError as e:
                print(e)
                
    # Compute and save means of f1-score, precision, recall, sensitivity, specificity
    total_path = experiment_dir / 'total'  # Directory where average metrics will be saved
    os.makedirs(total_path, exist_ok=True)
    ex_dirs = sorted(experiment_dir.glob('ex-*'))
    
    for base_name in ('case_nonr', 'pid_nonr'):
        df_scores = []
        score = 'accuracy'
        values = {}
        for ex_dir in ex_dirs:
            clrep_name = base_name + '_clrep.csv'
            clrep_path = ex_dir / clrep_name
            try:
                df_clrep = pd.read_csv(clrep_path, index_col=0)
                value = df_clrep.loc['precision'][score]
                values[str(ex_dir.name)] = value
            except FileNotFoundError as e:
                values[str(ex_dir.name)] = np.nan
                print(e)

        df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
    
        for score in ('f1-score', 'precision', 'recall'):
            values = {}
            for ex_dir in ex_dirs:
                clrep_name = base_name + '_clrep.csv'
                clrep_path = ex_dir / clrep_name
                try:
                    df_clrep = pd.read_csv(clrep_path, index_col=0)
                    value = df_clrep.loc[score][target_label]
                    values[str(ex_dir.name)] = value
                except FileNotFoundError as e:
                    values[str(ex_dir.name)] = np.nan
                    print(e)
            df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
            
        for score in ('sensitivity', 'specificity'):
            values = {}
            for ex_dir in ex_dirs:
                clrep_name = base_name + '_sensitivity_specificity.csv'
                clrep_path = ex_dir / clrep_name
                try:
                    df_clrep = pd.read_csv(clrep_path, index_col=0)
                    value = df_clrep.loc[score][target_label]
                    values[str(ex_dir.name)] = value
                except FileNotFoundError as e:
                    values[str(ex_dir.name)] = np.nan
                    print(e)
            df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
    
        # Aggregate metrics
        df_scores = pd.concat(df_scores, axis=1)
        print(base_name)
        scores_name = base_name + '_meanscores.csv'
        scores_path = total_path / scores_name
        df_scores.to_csv(scores_path)

        # Histograms
        fig = plt.figure(figsize=(25, 25))
        for i, col in enumerate(df_scores.columns):
            ax = plt.subplot2grid((5, 3), (i // 3, i % 3))
            sns.histplot(df_scores[col], ax=ax)
            ax.set_title(col)
        plt.show()
        
        # Descriptive statistics and max values
        df_describe = df_scores.describe()
        files = []
        for score in ('accuracy', 'f1-score', 'precision', 'recall', 'sensitivity', 'specificity'):
            filename = df_scores.loc[[df_scores[score].idxmax()]].index[0]
            files.append(filename)
        df_describe.loc['max file'] = files
        display(df_describe)
        describe_name = base_name + '_describe.csv'
        describe_path = total_path / describe_name
        df_describe.to_csv(describe_path) 
        
        max_recall_ex = df_scores.loc[[df_scores['recall'].idxmax()]].index[0]
        confmat_name = base_name + '_confmat.csv'
        confmat_path = experiment_dir / max_recall_ex / confmat_name
        df_confmat = pd.read_csv(confmat_path, index_col=0)
        display(df_confmat)


def plot_roc_from_files(directory, base_name, labels, inverse=False):
    """
    Load prediction probability and true/pred CSVs from directory,
    compute ROC curves per class and macro average,
    and save the ROC curve figure and CSVs for ROC and AUC.
    """
    fpr = {}
    tpr = {}
    roc_auc = {}
    precision = {}
    recall = {}
    pr_auc = {}

    n_classes = len(labels)
    pred_probs_name = base_name + '_pred_probs.csv'
    pred_probs_path = directory / pred_probs_name
    df_pred_probs = pd.read_csv(pred_probs_path, delimiter=',')

    true_pred_name = base_name + '_true_pred.csv'
    true_pred_path = directory / true_pred_name
    df_true_pred = pd.read_csv(true_pred_path, delimiter=',')

    y_test = df_true_pred['true'].values
    # One-hot encoding
    ohe = OneHotEncoder(sparse_output=False)
    y_test_one_hot = ohe.fit_transform(y_test.reshape(-1, 1))
    y_pred_proba = df_pred_probs.values

    # Per-class FPR, TPR, Precision, Recall
    for i in range(n_classes):
        if inverse is True:
            # Optionally swap true and predicted probabilities for some special analysis
            y_test_one_hot[:, i], y_pred_proba[:, i] = y_pred_proba[:, i], y_test_one_hot[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # ROC curve: compute macro average
    # Collect all unique FPR points
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Compute the mean TPR at each FPR
    mean_tpr = np.zeros_like(all_fpr)
    # Interpolate TPR for each class and sum them
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Take the average
    mean_tpr = mean_tpr / len(labels)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    colors = ['b', 'g', 'r']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Example: currently only plotting class 0
    for i in range(1):
        ax.plot(fpr[i], tpr[i], label=f'{labels[i]}', color=colors[i])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    fig_name = base_name + '_roc_curve.png'
    fig_path = directory / fig_name
    plt.savefig(fig_path)
    plt.show()

    # Print AUCs
    for i in range(n_classes):
        print(f'AUC({labels[i]}) =', roc_auc[i])
    print('AUC(mean) =', roc_auc["macro"])

    # Save ROC points for each class
    for i, label in enumerate(labels):
        roc_name = base_name + '_roc_' + label + '.csv'
        roc_path = directory / roc_name
        df_roc = pd.DataFrame(np.array([fpr[i], tpr[i]]).T, columns=('FPR', 'TPR'))
        df_roc.to_csv(roc_path, index=None)

    # Save macro-average ROC
    roc_name = base_name + '_roc_mean.csv'
    roc_path = directory / roc_name
    df_roc = pd.DataFrame(np.array([fpr['macro'], tpr['macro']]).T, columns=('FPR', 'TPR'))
    df_roc.to_csv(roc_path, index=None)

    # Save AUCs
    auc_name = base_name + '_roc_auc.csv'
    auc_path = directory / auc_name
    df_auc = pd.DataFrame(roc_auc.values(), index=tuple(labels) + ('mean',), columns=('AUC',))
    df_auc.to_csv(auc_path)


def plot_roc(df_pred_probs, df_true_pred, labels, colors=['b', 'g', 'r']):
    """
    Plot ROC curves from prediction probabilities and true labels DataFrames,
    and return the ROC DataFrames and AUC values.
    """
    fpr = {}
    tpr = {}
    roc_auc = {}
    precision = {}
    recall = {}
    pr_auc = {}

    n_classes = len(labels)

    y_test = df_true_pred['true'].values
    # One-hot encoding
    ohe = OneHotEncoder(sparse_output=False)
    y_test_one_hot = ohe.fit_transform(y_test.reshape(-1, 1))
    y_pred_proba = df_pred_probs.values

    # Per-class FPR, TPR, Precision, Recall
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # ROC macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr = mean_tpr / len(labels)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'{labels[i]}', color=colors[i])
    ax.plot(fpr['macro'], tpr['macro'], label='mean', color=colors[2])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()

    # Print AUCs
    for i in range(n_classes):
        print(f'AUC({labels[i]}) =', roc_auc[i])
    print('AUC(mean) =', roc_auc["macro"])

    # Prepare DataFrames
    # ROC per label
    df_roc_list = []
    for i, label in enumerate(labels):
        df_roc = pd.DataFrame(np.array([fpr[i], tpr[i]]).T, columns=('FPR', 'TPR'))
        df_roc_list.append(df_roc)
    # Macro-average ROC
    df_roc_mean = pd.DataFrame(np.array([fpr['macro'], tpr['macro']]).T, columns=('FPR', 'TPR'))
    # AUCs
    df_auc = pd.DataFrame(roc_auc.values(), index=tuple(labels) + ('mean',), columns=('AUC',))

    return df_roc_list, df_roc_mean, df_auc, plt


def plot_and_save_roc_from_files(directory, base_name, labels, colors=['b', 'g', 'r']):
    """
    Load prediction probabilities and true labels from files,
    plot ROC curves, save the figure and ROC/AUC CSVs.
    """
    n_classes = len(labels)

    pred_probs_name = base_name + '_pred_probs.csv'
    pred_probs_path = directory / pred_probs_name
    df_pred_probs = pd.read_csv(pred_probs_path, delimiter=',')

    true_pred_name = base_name + '_true_pred.csv'
    true_pred_path = directory / true_pred_name
    df_true_pred = pd.read_csv(true_pred_path, delimiter=',')

    df_roc_list, df_roc_mean, df_auc, plt_obj = plot_roc(df_pred_probs, df_true_pred, labels, colors)
    
    fig_name = base_name + '_roc_curve.png'
    fig_path = directory / fig_name
    plt_obj.savefig(fig_path)
    plt_obj.show()

    # Save ROC data
    for label, df_roc in zip(labels, df_roc_list):
        roc_name = base_name + '_roc_' + label + '.csv'
        roc_path = directory / roc_name
        df_roc.to_csv(roc_path, index=None)

    roc_name = base_name + '_roc_mean.csv'
    roc_path = directory / roc_name
    df_roc_mean.to_csv(roc_path, index=None)

    auc_name = base_name + '_roc_auc.csv'
    auc_path = directory / auc_name
    df_auc.to_csv(auc_path)


def pred_probsum_bypid(
    pids_test,   # PIDs in the test data
    z_test,      # PID for each instance in the test data
    y_pred_prob, # Predicted probabilities for each instance
    labels       # Label names
):
    """Compute the sum of predicted probabilities per PID and normalize to ratios."""
    pidlabels_pred_probs = []
    for pid in pids_test:  # Iterate over each PID in the test data
        y_pred_prob_bypid = y_pred_prob[z_test == pid]  # Predictions for this PID
        y_pred_prob_bypid_sum = y_pred_prob_bypid.sum(axis=0)
        pidlabels_pred_probs.append(y_pred_prob_bypid_sum)

    pidlabels_pred_probs = np.array(pidlabels_pred_probs)
    df_pidlabels_pred_prob = pd.DataFrame(pidlabels_pred_probs, columns=labels)
    # Convert to row-wise ratios
    df_pidlabels_pred_prob = df_pidlabels_pred_prob.apply(lambda x: x / x.sum(), axis=1)
    
    return df_pidlabels_pred_prob


def calc_save_ave_2class_maxf1(experiment_dir, labels, target_label='macro avg'):
    """
    Same as calc_save_ave_2class, but when selecting the "best" confusion matrix
    at the end, use the experiment with the maximum f1-score instead of recall.
    """

    # Compute and save sensitivity and specificity
    directories = sorted(experiment_dir.glob('ex-*'))
    
    for directory in directories:
        for base_name in ('case_nonr', 'pid_nonr'):
            confmat_name = base_name + '_confmat.csv'
            confmat_path = directory / confmat_name
            try:
                df_confmat = pd.read_csv(confmat_path, index_col=0)
                confmat = df_confmat.values
                df_sensitivity_specificity = sensitivity_specificity(confmat, labels)
                sensitivity_specificity_name = base_name + '_sensitivity_specificity.csv'
                sensitivity_specificity_path = directory / sensitivity_specificity_name
                df_sensitivity_specificity.to_csv(sensitivity_specificity_path)
            except FileNotFoundError as e:
                print(e)
                
    # Compute and save means of f1-score, precision, recall, sensitivity, specificity
    total_path = experiment_dir / 'total'  # Directory where average metrics will be saved
    os.makedirs(total_path, exist_ok=True)
    ex_dirs = sorted(experiment_dir.glob('ex-*'))
    
    for base_name in ('case_nonr', 'pid_nonr'):
        df_scores = []
        score = 'accuracy'
        values = {}
        for ex_dir in ex_dirs:
            clrep_name = base_name + '_clrep.csv'
            clrep_path = ex_dir / clrep_name
            try:
                df_clrep = pd.read_csv(clrep_path, index_col=0)
                value = df_clrep.loc['precision'][score]
                values[str(ex_dir.name)] = value
            except FileNotFoundError as e:
                values[str(ex_dir.name)] = np.nan
                print(e)

        df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
    
        for score in ('f1-score', 'precision', 'recall'):
            values = {}
            for ex_dir in ex_dirs:
                clrep_name = base_name + '_clrep.csv'
                clrep_path = ex_dir / clrep_name
                try:
                    df_clrep = pd.read_csv(clrep_path, index_col=0)
                    value = df_clrep.loc[score][target_label]
                    values[str(ex_dir.name)] = value
                except FileNotFoundError as e:
                    values[str(ex_dir.name)] = np.nan
                    print(e)
            df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
            
        for score in ('sensitivity', 'specificity'):
            values = {}
            for ex_dir in ex_dirs:
                clrep_name = base_name + '_sensitivity_specificity.csv'
                clrep_path = ex_dir / clrep_name
                try:
                    df_clrep = pd.read_csv(clrep_path, index_col=0)
                    value = df_clrep.loc[score][target_label]
                    values[str(ex_dir.name)] = value
                except FileNotFoundError as e:
                    values[str(ex_dir.name)] = np.nan
                    print(e)
            df_scores.append(pd.DataFrame(values.values(), index=values.keys(), columns=(score,)))
    
        # Aggregate metrics
        df_scores = pd.concat(df_scores, axis=1)
        print(base_name)
        scores_name = base_name + '_meanscores.csv'
        scores_path = total_path / scores_name
        df_scores.to_csv(scores_path)

        # Histograms
        fig = plt.figure(figsize=(25, 25))
        for i, col in enumerate(df_scores.columns):
            ax = plt.subplot2grid((5, 3), (i // 3, i % 3))
            sns.histplot(df_scores[col], ax=ax)
            ax.set_title(col)
        plt.show()
        
        # Descriptive statistics and max values
        df_describe = df_scores.describe()
        files = []
        for score in ('accuracy', 'f1-score', 'precision', 'recall', 'sensitivity', 'specificity'):
            filename = df_scores.loc[[df_scores[score].idxmax()]].index[0]
            files.append(filename)
        df_describe.loc['max file'] = files
        display(df_describe)
        describe_name = base_name + '_describe.csv'
        describe_path = total_path / describe_name
        df_describe.to_csv(describe_path) 
        
        # Use the experiment with maximum f1-score instead of recall
        max_recall_ex = df_scores.loc[[df_scores['f1-score'].idxmax()]].index[0]
        confmat_name = base_name + '_confmat.csv'
        confmat_path = experiment_dir / max_recall_ex / confmat_name
        df_confmat = pd.read_csv(confmat_path, index_col=0)
        display(df_confmat)
