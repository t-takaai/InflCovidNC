import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Kentai:
    '''Preparation of specimen data'''
    def __init__(self, featurepath, idlabelspath):
        self.features = pd.read_csv(featurepath,index_col=0)
        self.idlabels = pd.read_csv(idlabelspath,index_col=0)

        pid_count = self.idlabels.groupby(['pid']).count()
        if 'group' in self.idlabels.columns:
            label_pid_count = self.idlabels.groupby(['pid','label','group']).count()
        else:
            label_pid_count = self.idlabels.groupby(['pid','label']).count()
        label_pid_count['count']=pid_count.values[:,0]
        self.df_label_pid_count = label_pid_count.reset_index()

    def count_instance(self):
        # Count the number of instances per label
        df_label_count = pd.DataFrame(self.df_label_pid_count.groupby('label')['count'].sum())
        # Count the number of specimens (pids) per label
        df_lpc = pd.DataFrame(self.df_label_pid_count.groupby('label')['pid'].count())
        # Combine into a single DataFrame
        df_lpc['pulse']=self.df_label_pid_count.groupby('label')['count'].sum()
        return df_lpc
    
    def label_to_pids(self, label, label_column='label',pids_column='pid'):
        '''Return a numpy array of pids (possibly multiple) for specimens with the specified label.'''
        pids = np.array(list(self.idlabels[self.idlabels[label_column]==label].groupby(pids_column).groups.keys()))
        return pids

    def pids_to_labels(self, pids, label_column='label',pids_column='pid'):
        '''Return an array of labels corresponding to the given pid list.'''
        if np.isscalar(pids):  # If pids is a scalar, wrap it in a list
            pids = [pids]
        label_list = []
        for pid in pids:
            label = self.idlabels[self.idlabels['pid']==pid]['label'].iloc[0]
            label_list.append(label)
        return np.array(label_list)
    
    def split_pids(self,pids,train_size,test_size,random_state):
        '''Split the elements of pids (numpy array) into train and test sets.'''
        train_pids, test_pids\
            = train_test_split(pids,\
                               test_size=test_size,\
                               train_size=train_size,\
                               random_state=random_state)

        return train_pids, test_pids
    
    def split_pids_using_dict(self, n_pid_split_dict, random_state, label_column='label',\
                             pids_column='pid'):
        '''
        Using a dictionary that maps each label to [train_size, test_size]
        (e.g., {'CN2': [11, 10], 'CN6': [3, 3]}), split pids and
        return train_pids, test_pids along with their corresponding label lists.
        '''
        train_pids_list = []
        test_pids_list = []
        train_pidlabels_list = []
        test_pidlabels_list = []
        
        for label,splits in n_pid_split_dict.items():
            # From the label, obtain an ndarray of pids
            pids = self.label_to_pids(label,label_column,pids_column)
#            splits = n_pid_split_dict[label]
            
            train_pids, test_pids = self.split_pids(pids,*splits,random_state=random_state)
            train_pids_list.extend(train_pids)
            test_pids_list.extend(test_pids)
            train_pidlabels = [label]*len(train_pids)
            test_pidlabels = [label]*len(test_pids)
            train_pidlabels_list.extend(train_pidlabels)
            test_pidlabels_list.extend(test_pidlabels)

        return train_pids_list, test_pids_list, train_pidlabels_list, test_pidlabels_list            

    def pids_to_xyz(self, pids, label_column='label', pids_column='pid'):
        '''Extract X, y, and z from a list of pids. z is the pid for each instance.'''
        if np.isscalar(pids):  # If pids is a scalar, wrap it in a list
            pids = [pids]
        X_ = []
        y_ = []
        z_ = []
        for pid in pids:
            X = self.features[self.idlabels[pids_column]==pid].values
            y = self.idlabels[self.idlabels[pids_column]==pid][label_column].values
            z = self.idlabels[self.idlabels[pids_column]==pid][pids_column].values
            X_.append(X)
            y_.append(y)
            z_.append(z)

        X_arr = np.concatenate(X_)
        y_arr = np.concatenate(y_)
        z_arr = np.concatenate(z_)
        
        return X_arr,y_arr,z_arr

    def pids_to_dfxy(self, pids, label_column='label', pids_column='pid'):
        '''Extract df_X and df_y from a list of pids.'''
        if np.isscalar(pids):  # If pids is a scalar, wrap it in a list
            pids = [pids]
        X_ = []
        y_ = []
        for pid in pids:
            X = self.features[self.idlabels[pids_column]==pid]
            y = self.idlabels[self.idlabels[pids_column]==pid]
            X_.append(X)
            y_.append(y)
    
        df_X = pd.concat(X_)
        df_y = pd.concat(y_)
        
        return df_X, df_y

    def generate_dfxypidspidlabels(self, n_pid_split_dict, random_states, label_column='label',\
                                 pids_column='pid'):
        '''Generator that yields (df_X_train, df_X_test), (df_y_train, df_y_test),
        (train_pids_list, test_pids_list), and (train_pidlabels_list, test_pidlabels_list).'''
    
        for random_state in random_states:
            train_pids_list, test_pids_list,\
            train_pidlabels_list,  test_pidlabels_list =\
                self.split_pids_using_dict(n_pid_split_dict,random_state=random_state,\
                                          label_column=label_column,pids_column=pids_column)
    
            df_X_train, df_y_train = self.pids_to_dfxy(train_pids_list,label_column,pids_column)
            df_X_test, df_y_test = self.pids_to_dfxy(test_pids_list,label_column,pids_column)
    
            yield [df_X_train, df_X_test],[df_y_train, df_y_test],\
                    [train_pids_list, test_pids_list],\
                    [train_pidlabels_list, test_pidlabels_list]