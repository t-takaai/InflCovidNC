# -*- coding: utf-8 -*-
"""
calc_save_ave_3class.py

Aggregation script for 3-class classification (e.g., infl, covid, nc).
- For each experiment directory ex-*, read *_confmat.csv and *_clrep.csv,
  compute per-class sensitivity (TPR) and specificity (TNR) treating each class
  as the positive class, and also compute their macro averages, then save them.
- For the overall aggregation under total/, save a CSV that summarizes
  macro-average based scores (accuracy, f1, precision, recall, sensitivity,
  specificity) for each ex-*.
- Additionally, save a CSV (per-class) that lists per-class scores for each ex-*.
- It assumes that "accuracy" is a column in the classification report, and
  accepts both 'accuracy' and the misspelled 'accracy'.
  If neither can be found, accuracy is recomputed from the confusion matrix.

Expected input files (in each ex-* directory):
- pid_nonr_confmat.csv : Confusion matrix with rows=true labels, columns=predicted labels
- pid_nonr_clrep.csv   : CSV version of classification_report
                         Rows: ['precision','recall','f1-score','support']
                         Columns: each class name + ['macro avg','weighted avg',
                                   'accuracy' or 'accracy']

Output files:
- ex-*/ *_sensitivity_specificity.csv            : index=['sensitivity','specificity'],
                                                   columns=each class + ['macro avg']
- ex-*/ *_sensitivity_specificity_macro.csv      : same as above, but only the 'macro avg' column
- total/*_meanscores.csv                         : macro-average based metrics across ex-* 
                                                   (accuracy, f1, precision, recall, sensitivity, specificity)
- total/*_meanscores_macro.csv                   : same as above (kept for backward compatibility)
- total/*_meanscores_perclass.csv                : per-class scores (precision/recall/f1,
                                                   sensitivity/specificity) listed across ex-*
- total/*_describe.csv                           : descriptive statistics for macro-average scores
                                                   (with 'max file' row indicating which ex-* had the max)

Example usage:
>>> from pathlib import Path
>>> from calc_save_ave_3class import calc_save_ave_3class
>>> calc_save_ave_3class(Path("path/to/experiment"), labels=['infl','covid','nc'])
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _reorder_confmat(df_conf: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """
    Try to reorder rows and columns of df_conf to match the order of `labels`.
    If neither rows nor columns can be fully aligned, return df_conf as is.
    """
    try:
        if set(labels).issubset(df_conf.index) and set(labels).issubset(df_conf.columns):
            return df_conf.loc[list(labels), list(labels)]
        if set(labels).issubset(df_conf.index):
            return df_conf.loc[list(labels)]
        if set(labels).issubset(df_conf.columns):
            return df_conf.loc[:, list(labels)]
    except Exception:
        pass
    return df_conf


def sensitivity_specificity_multiclass(confmat: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """
    From a KxK confusion matrix (rows=true labels, columns=predicted labels),
    compute sensitivity (TPR, recall) and specificity (TNR) for each class i
    treated as the positive class.

    Returns
    -------
    pd.DataFrame
        index=['sensitivity','specificity'], columns=labels + ['macro avg']
    """
    conf = np.asarray(confmat, dtype=np.float64)
    K = conf.shape[0]
    if conf.shape[0] != conf.shape[1]:
        raise ValueError(f"confusion matrix must be square (got {conf.shape})")

    total = conf.sum()
    if total == 0:
        # If confusion matrix is all zeros, return NaNs and macro avg over them
        data = {lab: {"sensitivity": np.nan, "specificity": np.nan} for lab in labels}
        df = pd.DataFrame(data).T.T
        df["macro avg"] = df.mean(axis=1, skipna=True)
        return df

    sens_list, spec_list = [], []
    for i in range(K):
        TP = conf[i, i]
        FN = conf[i, :].sum() - TP
        FP = conf[:, i].sum() - TP
        TN = total - (TP + FN + FP)
        sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        sens_list.append(sens)
        spec_list.append(spec)

    df = pd.DataFrame(
        {"sensitivity": sens_list, "specificity": spec_list},
        index=list(labels)[:K],
    ).T
    df["macro avg"] = df.mean(axis=1, skipna=True)
    ordered_cols = list(labels)[:K] + ["macro avg"]
    return df.loc[:, ordered_cols]


def _safe_first_numeric(series_like: pd.Series) -> Optional[float]:
    """Return the first non-NaN numeric value in a Series-like object. If none, return None."""
    if series_like is None:
        return None
    s = pd.to_numeric(series_like, errors="coerce").dropna()
    return float(s.iloc[0]) if len(s) else None


def _extract_accuracy_from_clrep(df_clrep: pd.DataFrame) -> Optional[float]:
    """
    Extract accuracy from a classification_report CSV (df_clrep).
    - Assumes either 'accuracy' or 'accracy' exists (supports the misspelling).
    - Takes the first numeric value found in that row/column.
    - Returns None if it cannot be obtained (→ recompute from confusion matrix).
    """
    # Check if accuracy is represented as a column
    for cand in ("accuracy", "accracy"):
        if cand in df_clrep.columns:
            val = _safe_first_numeric(df_clrep[cand])
            if val is not None:
                return val
    # Or as an index
    for cand in ("accuracy", "accracy"):
        if cand in df_clrep.index:
            val = _safe_first_numeric(df_clrep.loc[cand])
            if val is not None:
                return val
    return None


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first column name found in `candidates` that exists in df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def calc_save_ave_3class(
    experiment_dir: Path | str,
    labels: Sequence[str],
    base_names: Sequence[str] = ("case_nonr", "pid_nonr"),
    target_label: str = "macro avg",
    save_hist: bool = False,
) -> None:
    """
    Main function to aggregate and save evaluation metrics for 3-class classification.
    """
    experiment_dir = Path(experiment_dir)

    # 1) For each experiment, compute sensitivity/specificity and save them
    directories = sorted(experiment_dir.glob("ex-*"))
    for directory in directories:
        for base_name in base_names:
            confmat_path = directory / f"{base_name}_confmat.csv"
            if not confmat_path.exists():
                print(f"[WARN] not found: {confmat_path}")
                continue
            try:
                df_confmat = pd.read_csv(confmat_path, index_col=0)
                df_confmat = _reorder_confmat(df_confmat, labels)
                ss_df = sensitivity_specificity_multiclass(
                    df_confmat.values, labels=list(df_confmat.columns)
                )
                ss_df.to_csv(directory / f"{base_name}_sensitivity_specificity.csv")
                ss_df.loc[:, ["macro avg"]].to_csv(
                    directory / f"{base_name}_sensitivity_specificity_macro.csv"
                )
            except Exception as e:
                print(f"[ERROR] {confmat_path}: {e}")

    # 2) Under total/ directory, save macro-based and per-class metrics
    total_path = experiment_dir / "total"
    os.makedirs(total_path, exist_ok=True)
    ex_dirs = sorted(experiment_dir.glob("ex-*"))

    # (A) Macro metrics (equivalent to the old *_meanscores.csv)
    for base_name in base_names:
        macro_scores_frames = []

        # accuracy
        values_acc = {}
        for ex_dir in ex_dirs:
            clrep_path = ex_dir / f"{base_name}_clrep.csv"
            conf_path = ex_dir / f"{base_name}_confmat.csv"
            acc_val = np.nan
            try:
                df_clrep = pd.read_csv(clrep_path, index_col=0)
                tmp = _extract_accuracy_from_clrep(df_clrep)
                if tmp is not None:
                    acc_val = float(tmp)
                elif conf_path.exists():
                    df_conf = pd.read_csv(conf_path, index_col=0)
                    acc_val = np.trace(df_conf.values) / df_conf.values.sum()
            except FileNotFoundError:
                # If clrep is missing, try to compute from confusion matrix
                if conf_path.exists():
                    df_conf = pd.read_csv(conf_path, index_col=0)
                    acc_val = np.trace(df_conf.values) / df_conf.values.sum()
            except Exception as e:
                print(f"[ERROR] accuracy read failed at {clrep_path}: {e}")
                if conf_path.exists():
                    try:
                        df_conf = pd.read_csv(conf_path, index_col=0)
                        acc_val = np.trace(df_conf.values) / df_conf.values.sum()
                    except Exception as e2:
                        print(f"[ERROR] accuracy recompute failed at {conf_path}: {e2}")
            values_acc[str(ex_dir.name)] = acc_val
        macro_scores_frames.append(
            pd.DataFrame(values_acc.values(), index=values_acc.keys(), columns=("accuracy",))
        )

        # f1/precision/recall (macro)
        macro_col_candidates = (target_label, "macro avg", "macro_avg", "macro average", "macro")
        for score in ("f1-score", "precision", "recall"):
            values = {}
            for ex_dir in ex_dirs:
                clrep_path = ex_dir / f"{base_name}_clrep.csv"
                try:
                    df_clrep = pd.read_csv(clrep_path, index_col=0)
                    col = _pick_column(df_clrep, macro_col_candidates)
                    values[str(ex_dir.name)] = (
                        float(df_clrep.loc[score, col])
                        if (col and score in df_clrep.index)
                        else np.nan
                    )
                except FileNotFoundError as e:
                    print(e)
                    values[str(ex_dir.name)] = np.nan
                except Exception as e:
                    print(f"[ERROR] {clrep_path}: {e}")
                    values[str(ex_dir.name)] = np.nan
            macro_scores_frames.append(
                pd.DataFrame(values.values(), index=values.keys(), columns=(score,))
            )

        # sensitivity/specificity (macro)
        for score in ("sensitivity", "specificity"):
            values = {}
            for ex_dir in ex_dirs:
                ss_path = ex_dir / f"{base_name}_sensitivity_specificity.csv"
                try:
                    df_ss = pd.read_csv(ss_path, index_col=0)
                    values[str(ex_dir.name)] = (
                        float(df_ss.loc[score, "macro avg"]) if score in df_ss.index else np.nan
                    )
                except FileNotFoundError as e:
                    print(e)
                    values[str(ex_dir.name)] = np.nan
                except Exception as e:
                    print(f"[ERROR] {ss_path}: {e}")
                    values[str(ex_dir.name)] = np.nan
            macro_scores_frames.append(
                pd.DataFrame(values.values(), index=values.keys(), columns=(score,))
            )

        df_macro_scores = pd.concat(macro_scores_frames, axis=1)
        # Old name kept for backward compatibility
        df_macro_scores.to_csv(total_path / f"{base_name}_meanscores.csv")
        # New name
        df_macro_scores.to_csv(total_path / f"{base_name}_meanscores_macro.csv")

        # Descriptive statistics
        if not df_macro_scores.empty:
            df_describe = df_macro_scores.describe()
            files = []
            for score in ("accuracy", "f1-score", "precision", "recall", "sensitivity", "specificity"):
                try:
                    files.append(df_macro_scores.loc[[df_macro_scores[score].idxmax()]].index[0])
                except Exception:
                    files.append("")
            df_describe.loc["max file"] = files
            df_describe.to_csv(total_path / f"{base_name}_describe.csv")

    # (B) Per-class metrics across ex-* (saved side by side)
    for base_name in base_names:
        frames_perclass = []
        # precision/recall/f1-score per class
        for score in ("precision", "recall", "f1-score"):
            values_per_label = {lab: {} for lab in labels}
            for ex_dir in ex_dirs:
                clrep_path = ex_dir / f"{base_name}_clrep.csv"
                try:
                    df_clrep = pd.read_csv(clrep_path, index_col=0)
                    for lab in labels:
                        val = (
                            df_clrep.loc[score, lab]
                            if (score in df_clrep.index and lab in df_clrep.columns)
                            else np.nan
                        )
                        values_per_label[lab][str(ex_dir.name)] = (
                            float(val) if pd.notna(val) else np.nan
                        )
                except FileNotFoundError as e:
                    print(e)
                    for lab in labels:
                        values_per_label[lab][str(ex_dir.name)] = np.nan
                except Exception as e:
                    print(f"[ERROR] {clrep_path}: {e}")
                    for lab in labels:
                        values_per_label[lab][str(ex_dir.name)] = np.nan
            for lab in labels:
                frames_perclass.append(
                    pd.Series(values_per_label[lab], name=f"{score}_{lab}").to_frame()
                )

        # sensitivity/specificity per class
        for score in ("sensitivity", "specificity"):
            values_per_label = {lab: {} for lab in labels}
            for ex_dir in ex_dirs:
                ss_path = ex_dir / f"{base_name}_sensitivity_specificity.csv"
                try:
                    df_ss = pd.read_csv(ss_path, index_col=0)
                    for lab in labels:
                        val = (
                            df_ss.loc[score, lab]
                            if (score in df_ss.index and lab in df_ss.columns)
                            else np.nan
                        )
                        values_per_label[lab][str(ex_dir.name)] = (
                            float(val) if pd.notna(val) else np.nan
                        )
                except FileNotFoundError as e:
                    print(e)
                    for lab in labels:
                        values_per_label[lab][str(ex_dir.name)] = np.nan
                except Exception as e:
                    print(f"[ERROR] {ss_path}: {e}")
                    for lab in labels:
                        values_per_label[lab][str(ex_dir.name)] = np.nan
            for lab in labels:
                frames_perclass.append(
                    pd.Series(values_per_label[lab], name=f"{score}_{lab}").to_frame()
                )

        # accuracy (reference) as a single column
        values_acc = {}
        for ex_dir in ex_dirs:
            clrep_path = ex_dir / f"{base_name}_clrep.csv"
            conf_path = ex_dir / f"{base_name}_confmat.csv"
            acc_val = np.nan
            try:
                df_clrep = pd.read_csv(clrep_path, index_col=0)
                tmp = _extract_accuracy_from_clrep(df_clrep)
                if tmp is not None:
                    acc_val = float(tmp)
                elif conf_path.exists():
                    df_conf = pd.read_csv(conf_path, index_col=0)
                    acc_val = np.trace(df_conf.values) / df_conf.values.sum()
            except Exception:
                if conf_path.exists():
                    df_conf = pd.read_csv(conf_path, index_col=0)
                    acc_val = np.trace(df_conf.values) / df_conf.values.sum()
            values_acc[str(ex_dir.name)] = acc_val
        frames_perclass.append(pd.Series(values_acc, name="accuracy").to_frame())

        if frames_perclass:
            pd.concat(frames_perclass, axis=1).to_csv(
                total_path / f"{base_name}_meanscores_perclass.csv"
            )


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) >= 3:
#         root = Path(sys.argv[1])
#         labs = sys.argv[2:]
#     else:
#         print("Usage: python calc_save_ave_3class.py <experiment_dir> <label1> <label2> <label3> [...]")
#         sys.exit(0)
#     calc_save_ave_3class(root, labels=labs, base_names=("case_nonr", "pid_nonr"), target_label="macro avg", save_hist=False)
