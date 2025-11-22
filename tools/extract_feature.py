import numpy as np
import pandas as pd
from pathlib import Path
from tsfresh import select_features
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
import lightgbm as lgb

def pulse_to_df(targetdir):
    '''
    Read all pulse files in the directory and output them as a DataFrame.
    While doing so, compute the corrected current values and add the pulse ID as a column.
    '''
    dfs = []
    number = 0  # Serial number of pulses
    pulsepaths = targetdir.glob('pulse_values-*.csv')  # Paths of pulse files in targetdir
    pulsepaths = sorted(list(pulsepaths))  # Sort pulse file paths in targetdir
    for pulsepath in pulsepaths:
        df = pd.read_csv(pulsepath)  # Load data
        column_names = df.columns  # Get column names: 0:'#', 1:'baseline(nA)', 2:'current(nA)', 3:'target range'
        idx = str(pulsepath).find('values-')  # Position of 'values-' in filename
        pid = str(pulsepath)[idx+7:idx+12]  # Use the 5 digits following 'values-' as the pulse ID
        df['パルス抽出結果ID'] = pid  # Add a column 'パルス抽出結果ID' with pid
        df['current'] = df[column_names[2]] - df[column_names[1]]  # Corrected current = current - baseline (no sign inversion)
        df[column_names[0]] += number  # Update pulse numbering
        number = df.iloc[-1]['#']  # Update current last pulse number

        dfs.append(df.values)

    df_combined_array = np.vstack(dfs)
    df_combined = pd.DataFrame(df_combined_array)
    df_combined.columns = ['#', 'ベースライン(nA)', '電流値(nA)', '対象範囲',
                           'パルス抽出結果ID', 'current']
    df_combined = df_combined.astype({
        '#': 'int64',
        'ベースライン(nA)': 'float64',
        '電流値(nA)': 'float64',
        '対象範囲': 'object',
        'パルス抽出結果ID': 'object',
        'current': 'float64'
    })
    
    return df_combined

def build_features(label_dir_dic):
    '''
    For each entry in label_dir_dic,
    read pulse files in the directory and extract features (Tsfresh),
    then output the corresponding pulse IDs (pid) and labels.
    '''
    features = []
    ids = []
    for label, targetdir in label_dir_dic.items():
        df = pulse_to_df(targetdir)
        df_grouped = df.groupby('#')
        
        # Extract the last unique pulse extraction result ID for each pulse
        ds_ids = df_grouped.apply(lambda d: d['パルス抽出結果ID'].unique()[-1])
        ds_ids.name = 'pid'
        
        df_ids = pd.DataFrame(ds_ids)
        df_ids['label'] = [label] * len(df_ids)  # Assign labels
        ids.append(df_ids)
        
        # Extract Tsfresh features only for rows whose range is 'o'
        feature = extract_features(
            df[df['対象範囲'] == 'o'],
            column_id='#',
            column_value='current'
        )
        features.append(feature)
        
    features = pd.concat(features, ignore_index=True)
    ids = pd.concat(ids)
    ids = ids.reset_index(drop=True)  # Reset index

    return features, ids
