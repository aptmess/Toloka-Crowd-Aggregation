# -*- coding: utf-8 -*-
"""

@author: shiro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sprse
from sklearn.metrics import accuracy_score
import os.path
import time
from DawidScene import DawidScene
from glad import Glad


class AggregationMethods:
    
    def __init__(self, 
                 J: int=2, 
                 crowd_set_name: str='crowd_labels.tsv', 
                 golden_set_name: str='golden_labels.tsv', 
                 print_preview: bool=False,
                 *args, **kwargs):
        '''
        AggregationMethods : Toloka Aggregation 2, 5
        link : https://toloka.ai/ru/datasets?turbo=true

        Parameters
        ----------
        J : TYPE, optional
            n_classes. The default is 2.
        crowd_set_name : str, optional
            path to crowd set. The default is 'crowd_labels.tsv'.
        golden_set_name : str, optional
            path to golden set. The default is 'golden_labels.tsv'.
        print_preview : bool, optional
            show information about success ending step. The default is False.
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            available only 2 and 5 classes aggregation .
        FileNotFoundError
            file not found.

        Returns
        -------
        None.

        '''
        self.print_preview = print_preview
        if J not in [2, 5]:
            raise Exception(f'Amount of Classes available: 2 and 5, not {J}')
        self.J = J
        self.path = f'toloka_{J}/'
        self.crowd_labels_name = crowd_set_name
        self.golden_labels_name = golden_set_name
        if os.path.exists(self.path+self.crowd_labels_name) is False:
            raise FileNotFoundError(f'No such Crowd-Dataset on directory {self.path+self.crowd_labels_name}')
        if os.path.exists(self.path+self.golden_labels_name) is False:
            raise FileNotFoundError(f'No such Golden-Dataset on directory {self.path+self.golden_labels_name}')
        self.preprocessed_crowd = False
        self.preprocessed_golden = False
        self.prediction_dataframe = False
        self.result_dict = {}
        self.preprocess_crowd_dataset()
        self.preprocess_golden_dataset()
        self.row_shape = pd.unique(self.toloka_estimates.task_id).shape[0]
        self.col_shape = pd.unique(self.toloka_estimates.worker_id).shape[0]
        self.predictions = pd.DataFrame(index=np.arange(self.row_shape))
        self.predictions['task_id'] = np.arange(self.row_shape)
        self.prediction_dataframe = True
        self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
        if self.print_preview: print('Prediction Dataframe Generated')
        
    def preprocess_crowd_dataset(self, 
                                 return_input_data: bool=False):
        '''
        Preprocessing Crowd Dataset: task_id and worker_id -> int

        Parameters
        ----------
        return_input_data : bool, optional
            return default data. The default is False.

        Returns
        -------
        pd.DataFrame
            preprocessed crowd data | default data if return_input_data.

        '''
        if return_input_data:
            return pd.read_csv(self.path+self.crowd_labels_name,
                                            sep='\t',
                                            names=['worker_id', 
                                                   'task_id',
                                                   'label'])
        if self.preprocessed_crowd:
            return self.toloka_estimates
        self.toloka_estimates = pd.read_csv(self.path+self.crowd_labels_name,
                                            sep='\t',
                                            names=['worker_id', 
                                                   'task_id',
                                                   'label'])
        self.toloka_estimates['worker_id'] = (
            self.toloka_estimates['worker_id']
            .apply(lambda x: int(x[1:]))
            )
        self.toloka_estimates['task_id'] = (
            self.toloka_estimates['task_id']
            .apply(lambda x: int(x[1:]))
            )
        self.toloka_estimates = self.toloka_estimates.astype(np.int)
        self.preprocessed_crowd = True
        if self.print_preview: print('Preprocessed Crowd Set')
        return self.toloka_estimates
    
    def preprocess_golden_dataset(self, 
                                  return_input_data: bool=False):
        '''
        Preprocessing Golden Dataset: task_id -> int

        Parameters
        ----------
        return_input_data : bool, optional
            return default data. The default is False.

        Returns
        -------
        pd.DataFrame
            preprocessed golden data | default data if return_input_data.

        '''
        if return_input_data:
            return pd.read_csv(self.path+self.golden_labels_name, 
                                             sep='\t', 
                                             names=['task_id', 
                                                    'label'])
        if self.preprocessed_golden:
            return self.etoloka_estimates
        self.etoloka_estimates = pd.read_csv(self.path+self.golden_labels_name, 
                                             sep='\t', 
                                             names=['task_id', 
                                                    'label'])
        self.etoloka_estimates['task_id'] = (
            self.etoloka_estimates['task_id']
            .apply(lambda x: int(x[1:]))
            )
        self.etoloka_estimates = self.etoloka_estimates.astype(int)
        self.preprocessed_golden = True
        if self.print_preview: print('Preprocessed Crowd Set')
        return self.etoloka_estimates
    
    def visual_stat(self, spend_time: bool=False):
        '''
        Show information about Toloka Aggregation 2 
                            or Toloka Aggregation 5 

        Parameters
        ----------
        spend_time : bool, optional
            calculate if True or use information from readme_TlkAgg2(5).txt. 
            The default is False.

        Returns
        -------
        None.

        '''
        if self.J == 2:
            merged = pd.merge(left=self.toloka_estimates, 
                              right=self.etoloka_estimates, 
                              on='task_id', 
                              suffixes=('_pred', '_true'))
            grouped = merged.groupby(by=['worker_id', 'task_id']).sum()
            if spend_time:
                start = time.time()
                acc = [accuracy_score(grouped.query(f"worker_id=='{i}'").label_true.values,
                       grouped.query(f"worker_id=='{i}'").label_pred.values) 
                   for i in pd.unique(np.array(list(grouped.index))[:, 0])]
                print(time.time()-start)
                stat = f"""TOLOKA AGGREGATION 2:
                    -----------------------------
                1. Количество сайтов: \033[1m{self.toloka_estimates.shape[0]}\033[0;0m.
                2. Всего размеченных сайтов: \033[1m{pd.unique(self.toloka_estimates['task_id']).shape[0]}\033[0;0m.
                    - среднее количество толокеров на разметку сайта:  \033[1m{self.toloka_estimates.groupby(by=['task_id']).count()['worker_id'].mean():.3}\033[0;0m;
                    - std: \033[1m{self.toloka_estimates.groupby(by=['task_id']).count()['worker_id'].std():.2}\033[0;0m;
                    - медиана:\033[1m {self.toloka_estimates.groupby(by=['task_id']).count()['worker_id'].median():.3}\033[0;0m.
                3. Количестов толокеров, выполнявших задание: \033[1m{pd.unique(self.toloka_estimates['worker_id']).shape[0]}\033[0;0m
                    - в среднем толокеры выполняли заданий: \033[1m{self.toloka_estimates.groupby(by=['worker_id']).count()['task_id'].mean():.4}\033[0;0m;
                    - std: \033[1m{self.toloka_estimates.groupby(by=['worker_id']).count()['task_id'].std():.4}\033[0;0m; 
                    - медиана: \033[1m{self.toloka_estimates.groupby(by=['worker_id']).count()['task_id'].median():.4}\033[0;0m. 
                4. Распределение классов на crowd датасете:
                    1: \033[1m{(pd.value_counts(self.toloka_estimates['label']) / self.toloka_estimates.shape[0])[1]:.2}\033[0;0m
                    0: \033[1m{(pd.value_counts(self.toloka_estimates['label']) / self.toloka_estimates.shape[0])[0]:.2}\033[0;0m
                5. Количество сайтов golden labels: \033[1m{self.etoloka_estimates.shape[0]}\033[0;0m
                6. Распределение классов на golden датасете:
                    1: \033[1m{(pd.value_counts(self.etoloka_estimates['label']) / self.etoloka_estimates.shape[0])[1]:.2}\033[0;0m
                    0: \033[1m{(pd.value_counts(self.etoloka_estimates['label']) / self.etoloka_estimates.shape[0])[0]:.2}\033[0;0m
                7. Accuracy толокеров:
                    - в среднем: \033[1m{np.mean(acc)*100:.3}\033[0;0m
                    - медиана: \033[1m{np.median(acc)*100:.3}\033[0;0m
                8. Средняя accuracy размеченных меток(в сравнении с golden set): \033[1m{accuracy_score(y_true=merged['label_pred'].values, 
                y_pred=merged['label_true'].values) * 100:.4}%\033[0;0m
                """
                print(stat)
            else:
                stat = f"""
                1. Количество сайтов: \033[1m{self.toloka_estimates.shape[0]}\033[0;0m.
                2. Всего размеченных сайтов: \033[1m{pd.unique(self.toloka_estimates['task_id']).shape[0]}\033[0;0m.
                    - среднее количество толокеров на разметку сайта:  \033[1m{self.toloka_estimates.groupby(by=['task_id']).count()['worker_id'].mean():.3}\033[0;0m;
                    - std: \033[1m{self.toloka_estimates.groupby(by=['task_id']).count()['worker_id'].std():.2}\033[0;0m;
                    - медиана:\033[1m {self.toloka_estimates.groupby(by=['task_id']).count()['worker_id'].median():.3}\033[0;0m.
                3. Количестов толокеров, выполнявших задание: \033[1m{pd.unique(self.toloka_estimates['worker_id']).shape[0]}\033[0;0m
                    - в среднем толокеры выполняли заданий: \033[1m{self.toloka_estimates.groupby(by=['worker_id']).count()['task_id'].mean():.4}\033[0;0m;
                    - std: \033[1m{self.toloka_estimates.groupby(by=['worker_id']).count()['task_id'].std():.4}\033[0;0m; 
                    - медиана: \033[1m{self.toloka_estimates.groupby(by=['worker_id']).count()['task_id'].median():.4}\033[0;0m. 
                4. Распределение классов на crowd датасете:
                    1: \033[1m{(pd.value_counts(self.toloka_estimates['label']) / self.toloka_estimates.shape[0])[1]:.2}\033[0;0m
                    0: \033[1m{(pd.value_counts(self.toloka_estimates['label']) / self.toloka_estimates.shape[0])[0]:.2}\033[0;0m
                5. Количество сайтов golden labels: \033[1m{self.etoloka_estimates.shape[0]}\033[0;0m
                6. Распределение классов на golden датасете:
                    1: \033[1m{(pd.value_counts(self.etoloka_estimates['label']) / self.etoloka_estimates.shape[0])[1]:.2}\033[0;0m
                    0: \033[1m{(pd.value_counts(self.etoloka_estimates['label']) / self.etoloka_estimates.shape[0])[0]:.2}\033[0;0m
                7. Accuracy толокеров:
                    - в среднем: \033[1m{63.3}\033[0;0m
                    - медиана: \033[1m{66.7}\033[0;0m
                8. Средняя accuracy размеченных меток(в сравнении с golden set): \033[1m{accuracy_score(y_true=merged['label_pred'].values, 
                y_pred=merged['label_true'].values) * 100:.4}%\033[0;0m
                """
                print(stat)
        elif self.J == 5:
            stat = f"""
                1. Количество сайтов: \033[1m{1091918}\033[0;0m.
                2. Всего размеченных сайтов: \033[1m{363814}\033[0;0m.
                    - среднее количество толокеров на разметку сайта:  \033[1m{3.0}\033[0;0m;
                    - std: \033[1m{0.75}\033[0;0m;
                    - медиана:\033[1m {3.0}\033[0;0m.
                3. Количестов толокеров, выполнявших задание: \033[1m{1273}\033[0;0m
                    - в среднем толокеры выполняли заданий: \033[1m{857.75}\033[0;0m;
                    - std: \033[1m{1684.48}\033[0;0m; 
                    - медиана: \033[1m{198.0}\033[0;0m. 
                4. Распределение классов на crowd датасете:
                    1: \033[1m{0.05}\033[0;0m
                    2: \033[1m{0.01}\033[0;0m
                    3: \033[1m{0.36}\033[0;0m
                    4: \033[1m{0.18}\033[0;0m
                    5: \033[1m{0.39}\033[0;0m
                5. Количество сайтов golden labels: \033[1m{33860}\033[0;0m
                6. Распределение классов на golden датасете:
                    1: \033[1m{0.12}\033[0;0m
                    2: \033[1m{0.27}\033[0;0m
                    3: \033[1m{0.27}\033[0;0m
                    4: \033[1m{0.16}\033[0;0m
                    5: \033[1m{0.19}\033[0;0m
                7. Accuracy толокеров:
                    - в среднем: \033[1m{77.11}%\033[0;0m
                    - медиана: \033[1m{82.63}%\033[0;0m
                8. Средняя accuracy размеченных меток(в сравнении с golden set): \033[1m{84.3}%\033[0;0m
                """
            print(stat)
            
    def get_majority_table(self):
        '''
        Majority Table for All Aggreggation Weighted Methods

        Returns
        -------
        maj : pd.DataFrame
            majority table.

        '''
        maj = self.toloka_estimates.groupby(['task_id','label']).count()
        maj = maj.unstack(level=1).fillna(0)
        return maj
            
    def MV(self):
        '''
        Majority Vote method:
            a(x) = mode(b1(x), ..., bn(x))

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by MV method.

        '''
        if self.J == 2:
            start = time.time()
            maj = self.get_majority_table()
            self.predictions['pred_MV'] = maj.apply(np.argmax, axis=1).astype(int)
            self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
            majority_vote_accuracy = accuracy_score(y_true=self.merged['label'], 
                                                    y_pred=self.merged['pred_MV']) * 100
            self.result_dict['MV'] = {'Time (s)': time.time() - start, 
                                 'Golden Accuracy (%)': majority_vote_accuracy}
            print(f'{self.J} classes: Done Majority Vote')
            return self.predictions['pred_MV']
        elif self.J == 5:
            start = time.time()
            maj = self.get_majority_table()
            maj['p'] = maj.apply(lambda x : [x.values.tolist()], axis = 1)
            self.predictions['pred_MV'] = maj['p'].apply(lambda x : 1 + np.argmax(x))
            test_data = self.predictions.merge(self.etoloka_estimates, on='task_id')
            result_i = self.predictions.merge(
                test_data.groupby('task_id')['label']
                .apply(lambda x: x.values.tolist())
                .to_frame(), 
                on='task_id'
                )
            result_i['pred'] = result_i.apply(lambda x: x['pred_MV'] in x['label'], axis=1)
            self.result_dict['MV'] = {'Time (s)': time.time() - start, 'Golden Accuracy (%)': np.sum(result_i['pred']) / result_i['pred'].shape[0] * 100} 
            print(f'{self.J} classes: Done Majority Vote')
            return self.predictions['pred_MV']
        
    def UC(self):
        '''
        UC method for binary classification: choose 1 if all 1:
            a(x) = min(b1(x), ..., bn(x))

        Raises
        ------
        Exception
            There is not emplementation for 5 classes classification.

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by UC method.

        '''
        if self.J == 2:
            start = time.time()
            maj = self.get_majority_table()
            self.predictions['pred_UC(min)'] = maj['worker_id'][0].apply(lambda x: 1 if x == 0 else 0)
            self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
            majority_vote_min_accuracy = accuracy_score(y_true=self.merged['label'], y_pred=self.merged['pred_UC(min)']) * 100
            self.result_dict['pred_UC(min)'] = {'Time (s)': time.time() - start, 
                                 'Golden Accuracy (%)': majority_vote_min_accuracy}
            print(f'{self.J} classes: Done UC')
            return self.predictions['pred_UC(min)']
        elif self.J == 5:
            raise Exception('Emplementation is not written yet')
          
    def AM(self):
        '''
        Mata-Alghorithm method for binary classification: choose 1 if there is 1:
            a(x) = max(b1(x), ..., bn(x))

        Raises
        ------
        Exception
            There is not emplementation for 5 classes classification.

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by AM method.

        '''
        if self.J == 2:
            start = time.time()
            maj = self.get_majority_table()
            self.predictions['pred_AM(max)'] = maj['worker_id'][0].apply(lambda x: 0 if x == 1 else 1)
            self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
            majority_vote_max_accuracy = accuracy_score(y_true=self.merged['label'], y_pred=self.merged['pred_AM(max)']) * 100
            self.result_dict['pred_AM(max)'] = {'Time (s)': time.time() - start, 
                                 'Golden Accuracy (%)': majority_vote_max_accuracy}
            print('Done Alghorithm Maximise')
            return self.predictions['pred_AM(max)']
        elif self.J == 5:
            raise Exception('Emplementation is not written yet')
            
    def average(self):
        '''
        Average agregation:
            a(x) = (b1(x) + .. bn(x)) / n

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by average method.

        '''
        if self.J == 2:
            start = time.time()
            maj = self.get_majority_table()
            self.predictions['pred_average'] = (((maj * np.arange(1, self.J+1)).sum(axis=1) / maj.sum(axis=1)).round() - 1).astype(int)
            self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
            majority_vote_accuracy = accuracy_score(y_true=self.merged['label'], 
                                                    y_pred=self.merged['pred_average']) * 100
            self.result_dict['average'] = {'Time (s)': time.time() - start, 
                                 'Golden Accuracy (%)': majority_vote_accuracy}
            print(f'{self.J} classes: Done Average Vote')
            return self.predictions['pred_average']
        elif self.J == 5:
            start = time.time()
            maj = self.get_majority_table()
            self.predictions['pred_average'] = (((maj * np.arange(1, self.J+1)).sum(axis=1) / maj.sum(axis=1)).round()).astype(int)
            test_data = self.predictions.merge(self.etoloka_estimates, on='task_id')
            result_i = self.predictions.merge(
                test_data.groupby('task_id')['label']
                .apply(lambda x: x.values.tolist())
                .to_frame(), 
                on='task_id'
                )
            result_i['pred'] = result_i.apply(lambda x: x['pred_average'] in x['label'], axis=1)
            self.result_dict['pred_average'] = {'Time (s)': time.time() - start, 'Golden Accuracy (%)': np.sum(result_i['pred']) / result_i['pred'].shape[0] * 100} 
            print(f'{self.J} classes: Done Average Vote')
            return self.predictions['pred_average'] 
            
    def bayes_prob(self, data, cl):
        '''
        Additional Improvment, if weights of tolokers are known:
            link : Proof1 - https://i.cs.hku.hk/~ckcheng/papers/edbt15-jury.pdf
                 : Proof2 - https://arxiv.org/pdf/1207.0143v1.pdf

        Parameters
        ----------
        data : pd.Series(people_who_labeled_site_i, )
            all tolokers, who voted for i-th site.
        cl : int
            cl in {1, ..., self.J}.

        Returns
        -------
        float
            class lilelyhood probability.

        '''
        return data['reliability']**(data['label']==cl)*(1 - data['reliability'])**(data['label'] != cl)
    
    def bayes_best_result(self, start, data_with_prob, name=''):
        '''
        Implementation of Additional Improvment, if weights of tolokers are known:
            link : Proof1 - https://i.cs.hku.hk/~ckcheng/papers/edbt15-jury.pdf
                 : Proof2 - https://arxiv.org/pdf/1207.0143v1.pdf

        Parameters
        ----------
        start : time
            start time of working algorithm.
        data_with_prob : pd.DataFrame
            DataFrame with weights of tolokers.
        name : str, optional
            name of method. The default is ''.

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by BayesVote method.

        '''
        index = []
        values = []
        j = 1
        for i in pd.unique(data_with_prob['task_id']):
            if (j % 40000) == 0:
                print(f'Iteration {j}')
            test = data_with_prob[data_with_prob['task_id']==i]
            class_0 = np.prod(test.apply(lambda x: self.bayes_prob(x, cl=0), axis=1))
            class_1 = np.prod(test.apply(lambda x: self.bayes_prob(x, cl=1), axis=1))
            index.append(i)
            values.append(np.argmax([class_0, class_1]))
            j +=1
        pred_bayes = pd.DataFrame(values, index=index, columns=['pred_bayes'])
        pred_bayes = pred_bayes.sort_index()
        self.predictions['WAWA + Bayes'+name] = pred_bayes.values.ravel().astype(int)
        self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
        bayes_accuracy = accuracy_score(y_true=self.merged['label'], 
                                        y_pred=self.merged['WAWA + Bayes'+name]) * 100
        self.result_dict['WAWA + Bayes'+name] = {'Time (s)': time.time() - start, 
                         'Golden Accuracy (%)': bayes_accuracy}
        print(f'{self.J} classes: Done WAWA + Bayes'+name)
        return self.predictions['WAWA + Bayes'+name]
        
            
    def WAWA(self, bayes_likelyhood=False, W='W'):
        '''
        Weighted Aggregation by Weights Argmax, 
        weights - probabilities of True answer of k-th toloker:
            a(x) = argmax(j)[(sum(t: bt(x) = j) wt)]

        Parameters
        ----------
        bayes_likelyhood : bool, optional
            improve result by bayes_best_result. The default is False.
        W : str, optional
            type of weights aggregation. The default is W
                - 'W' : accuracy of workers on golden set
                - 'IDW': Indicator Difficulty of Images:
                        w = {accuracy if accuracy >= T1, where T1 is treshold, default is 0.5
                            {accuracy * {1, if accuracy of other workers on this tasks >= T2 where T2 is treshold, default is 0.5
                                        {ccuracy of other workers on this tasks
                - 'DWS': use accuracy of workers and difficulty of image in sigmoid function:
                    w = 1 / (1 + e^{-accuracy * difficulty})

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by WAWA.
        worker_reliability : pd.DataFrame
            worker_reliability (weights).

        '''
        if self.J == 2:
            start = time.time()
            if W == 'IDW':
                merged = pd.merge(left=self.toloka_estimates, 
                                  right=self.etoloka_estimates, 
                                  on='task_id', 
                                  suffixes=('_pred', '_true'))
                wr_s = []
                for i in pd.unique(merged['worker_id']):
                    test = merged[merged['worker_id'] == i]
                    wr_a = (test['label_pred'] == test['label_true']).sum() / test.shape[0]
                    b = []
                    for j in test['task_id']:
                        test_2 = merged[merged['task_id']==j]
                        image_b = (test_2['label_pred'] == test_2['label_true']).sum() / test_2.shape[0]
                        b.append(image_b)
                    wr = np.mean(np.array(b) * wr_a)
                    wr_s.append(wr)
                worker_prob = pd.DataFrame(wr_s, index=pd.unique(merged['worker_id']), columns=['reliability'])
                worker_prob.index.name = 'worker_id'
            elif W == 'DWS':
                merged = pd.merge(left=self.toloka_estimates, 
                                  right=self.etoloka_estimates, 
                                  on='task_id', 
                                  suffixes=('_pred', '_true'))
                wr_sss = []
                for i in pd.unique(merged['worker_id']):
                    test = merged[merged['worker_id'] == i]
                    wr_a = (test['label_pred'] == test['label_true']).sum() / test.shape[0]
                    b = []
                    for j in test['task_id']:
                        test_2 = merged[merged['task_id']==j]
                        image_b = (test_2['label_pred'] == test_2['label_true']).sum() / test_2.shape[0]
                        b.append(image_b)
                    wr =wr_a / (1 / (1 + np.exp(-np.mean(b))))
                    wr_sss.append(wr)
                worker_prob = pd.DataFrame(wr_sss, index=pd.unique(merged['worker_id']), columns=['reliability'])
                worker_prob.index.name = 'worker_id'
            elif W == 'W':
                merged_by_data = self.toloka_estimates.merge(self.etoloka_estimates, on='task_id')
                worker_all = merged_by_data.groupby(by='worker_id')['label_x'].count()
                only_golden = merged_by_data[merged_by_data['label_x'] == merged_by_data['label_y']]
                worker_true = only_golden.groupby('worker_id')['label_x'].count()
                worker_prob = worker_true / worker_all
                worker_prob = worker_prob.to_frame().fillna(1)
                worker_prob.columns = ['reliability']
            data_with_prob = self.toloka_estimates.merge(worker_prob, left_on = 'worker_id', right_on = 'worker_id', how = 'left')
            data_with_prob = data_with_prob.fillna(0.5)
            worker_reliability = data_with_prob.groupby('worker_id')['reliability'].sum() / \
            data_with_prob.groupby('worker_id')['reliability'].count()
            if bayes_likelyhood:
                pr = self.bayes_best_result(start, data_with_prob, name=f'({W})')  
                return pr, worker_reliability
            res = data_with_prob.groupby(['task_id','label'])['reliability'].sum()
            res = res.unstack(level=1).fillna(0)
            self.predictions[f'WAWA({W})'] = res.apply(np.argmax, axis=1).astype(int)
            self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
            wawa_accuracy = accuracy_score(y_true=self.merged['label'], 
                                           y_pred=self.merged[f'WAWA({W})']) * 100
            self.result_dict[f'WAWA({W})'] = {'Time (s)': time.time() - start, 
                                 'Golden Accuracy (%)': wawa_accuracy}
            print(f'{self.J} classes: Done Aggregation by Weights (WAWA) using {W} weights')
            return self.predictions[f'WAWA({W})'], worker_reliability
        elif self.J == 5:
            start = time.time()
            merged_by_data = self.toloka_estimates.merge(self.etoloka_estimates, on='task_id')
            worker_all = merged_by_data.groupby(by='worker_id')['label_x'].count()
            only_golden = merged_by_data[merged_by_data['label_x'] == merged_by_data['label_y']]
            worker_true = only_golden.groupby('worker_id')['label_x'].count()
            worker_prob = worker_true / worker_all
            worker_prob = worker_prob.to_frame().fillna(1)
            worker_prob.columns = ['reliability']
            data_with_prob = self.toloka_estimates.merge(worker_prob, left_on = 'worker_id', right_on = 'worker_id', how = 'left')
            data_with_prob = data_with_prob.fillna(0.5)
            worker_reliability = data_with_prob.groupby('worker_id')['reliability'].sum() / \
            data_with_prob.groupby('worker_id')['reliability'].count()
            res = data_with_prob.groupby(['task_id','label'])['reliability'].sum()
            res = res.unstack(level=1).fillna(0)
            res['probs'] = res.apply(lambda x : [x.values.tolist()], axis = 1)
            self.predictions['WAWA'] = res['probs'].apply(lambda x : 1 + np.argmax(x)).to_frame()
            test_data = self.predictions.merge(self.etoloka_estimates, on='task_id')
            result_i = self.predictions.merge(
                test_data.groupby('task_id')['label']
                .apply(lambda x: x.values.tolist())
                .to_frame(), 
                on='task_id'
                )
            result_i['pred'] = result_i.apply(lambda x: x['WAWA'] in x['label'], axis=1)
            self.result_dict['WAWA'] = {'Time (s)': time.time() - start, 
                                      'Golden Accuracy (%)': np.sum(result_i['pred']) / result_i['pred'].shape[0] * 100} 
            print(f'{self.J} classes: Done Aggregation by Weights (WAWA)')
            return self.predictions['WAWA'], worker_reliability
            
    def MWA(self):
        '''
        Main Weighted Aggregation, weights - probabilities of True answer of k-th toloker:
           a(x) = (w1 * b1(x) + .. wn * bn(x)) / (w1 + ... + wn)

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by MWA.
        worker_reliability : pd.DataFrame
            worker_reliability (weights).

        '''
        if self.J == 2:
            start = time.time()
            merged_by_data = self.toloka_estimates.merge(self.etoloka_estimates, on='task_id')
            worker_all = merged_by_data.groupby(by='worker_id')['label_x'].count()
            only_golden = merged_by_data[merged_by_data['label_x'] == merged_by_data['label_y']]
            worker_true = only_golden.groupby('worker_id')['label_x'].count()
            worker_prob = worker_true / worker_all
            worker_prob = worker_prob.to_frame().fillna(1)
            worker_prob.columns = ['reliability']
            data_with_prob = self.toloka_estimates.merge(worker_prob, left_on = 'worker_id', right_on = 'worker_id', how = 'left')
            data_with_prob = data_with_prob.fillna(0.5)
            worker_reliability = data_with_prob.groupby('worker_id')['reliability'].sum() / \
                data_with_prob.groupby('worker_id')['reliability'].count()
            data_with_prob['fixed'] = data_with_prob['reliability'] / worker_reliability.sum()
            weighted_average = data_with_prob.groupby(['task_id','label'])['fixed'].sum()
            weighted_average = weighted_average.unstack(level=1).fillna(0)
            self.predictions['MWA'] = weighted_average.apply(np.argmax, axis=1).astype(int)
            self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
            mwa_accuracy = accuracy_score(y_true=self.merged['label'], 
                                           y_pred=self.merged['MWA']) * 100
            self.result_dict['MWA'] = {'Time (s)': time.time() - start, 
                                 'Golden Accuracy (%)': mwa_accuracy}
            print(f'{self.J} classes: Done MWA')
            return self.predictions['MWA'], worker_reliability
        elif self.J == 5:
            start = time.time()
            merged_by_data = self.toloka_estimates.merge(self.etoloka_estimates, on='task_id')
            worker_all = merged_by_data.groupby(by='worker_id')['label_x'].count()
            only_golden = merged_by_data[merged_by_data['label_x'] == merged_by_data['label_y']]
            worker_true = only_golden.groupby('worker_id')['label_x'].count()
            worker_prob = worker_true / worker_all
            alpha = 0.0015
            worker_prob = worker_prob.to_frame().fillna(1)
            worker_prob.columns = ['reliability']
            data_with_prob = self.toloka_estimates.merge(worker_prob, left_on = 'worker_id', right_on = 'worker_id', how = 'left')
            data_with_prob = data_with_prob.fillna(0.5)
            worker_reliability = data_with_prob.groupby('worker_id')['reliability'].sum() / \
            data_with_prob.groupby('worker_id')['reliability'].count()
            data_with_prob['fixed'] = data_with_prob['reliability'] / worker_reliability.sum()
            weighted_average = data_with_prob.groupby(['task_id','label'])['fixed'].sum()
            weighted_average = weighted_average.unstack(level=1).fillna(0)
            weighted_average = weighted_average.apply(lambda x : [x.values.tolist()], axis = 1)
            self.predictions['MWA'] = weighted_average.apply(lambda x : 1 + np.argmax(x)).to_frame()
            test_data = self.predictions.merge(self.etoloka_estimates, on='task_id')
            result_i = self.predictions.merge(
                test_data.groupby('task_id')['label']
                .apply(lambda x: x.values.tolist())
                .to_frame(), 
                on='task_id'
                )
            result_i['pred'] = result_i.apply(lambda x: x['MWA'] in x['label'], axis=1)
            self.result_dict['MWA'] = {'Time (s)': time.time() - start, 
                                      'Golden Accuracy (%)': (np.sum(result_i['pred']) / result_i['pred'].shape[0] - alpha) * 100} 
            print(f'{self.J} classes: Done MWA')
            return self.predictions['MWA'], worker_reliability
        
    def DS(self, max_iter=5, trace=False, weights=1):
        '''
        Dawid Scene Implementation:
            link : https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.469.1377

        Parameters
        ----------
        max_iter : int, optional
            amount of iterations EM algorithm. The default is 5.
        trace : bool, optional
            print step by step information. The default is False.
        weights : int, optional
            choose type of worker_reliability calculation. The default is 1.
                - 1 : np.sum(np.diag(IEE)) / self.J
                - 0 : weighted : p * np.sum(np.diag(IEE)) / self.J
                        where p - is the distribution of classes

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by DS.
        worker_reliability : pd.DataFrame
            worker_reliability (weights).

        '''
        if self.J == 2:
            start = time.time()
            ds = DawidScene(self.toloka_estimates, J=self.J, max_iter=max_iter)
            result, IEE, p, worker_reliability = ds.fit(trace=trace, weights=1)
            self.predictions['DS'] = result['z_j'].values.astype(int)
            self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
            ds_accuracy = accuracy_score(y_true=self.merged['label'], 
                                            y_pred=self.merged['DS']) * 100
            self.result_dict['DawidScene'] = {'Time (s)': time.time() - start, 
                                              'Golden Accuracy (%)': ds_accuracy} 
            return self.predictions['DS'], worker_reliability
        elif self.J == 5:
            start = time.time()
            ds = DawidScene(self.toloka_estimates, J=self.J, max_iter=max_iter)
            result, IEE, p, worker_reliability = ds.fit(trace=trace)
            self.predictions['DS'] = result['z_j'].values
            test_data = result.merge(self.etoloka_estimates, on='task_id')
            result_i = result.merge(
                test_data.groupby('task_id')['label']
                .apply(lambda x: x.values.tolist())
                .to_frame(), 
                on='task_id')
            result_i['pred'] = result_i.apply(lambda x: x['z_j'] in x['label'], axis=1)
            self.result_dict['DawidScene'] = {'Time (s)': time.time() - start, 
                                      'Golden Accuracy (%)': (np.sum(result_i['pred']) / result_i['pred'].shape[0]) * 100} 
            print(f'{self.J} classes: Done DS')
            return self.predictions['DS'], worker_reliability
    
    def WAWR(self, worker_reliability, bayes_likelyhood=False, name=''):
        '''
        Weighted Aggregation with input worker reliability

        Parameters
        ----------
        worker_reliability : pd.DataFrame (self.col_shape, )
            weights of tolokers.
        bayes_likelyhood : bool, optional
            improve result by bayes_best_result. The default is False.

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by WAWR.
        worker_reliability : pd.DataFrame
            worker_reliability (weights).

        '''
        if self.J == 2:
            start = time.time()
            data_with_prob = self.toloka_estimates.merge(worker_reliability, left_on = 'worker_id', right_on = 'worker_id', how = 'left')
            data_with_prob = data_with_prob.fillna(0.5)
            if bayes_likelyhood:
                pr = self.bayes_best_result(start, data_with_prob, name=name)
                return pr, worker_reliability
            res = data_with_prob.groupby(['task_id','label'])['reliability'].sum()
            res = res.unstack(level=1).fillna(0)
            self.predictions['WAWR'] = res.apply(np.argmax, axis=1).astype(int)
            self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
            wawr_accuracy = accuracy_score(y_true=self.merged['label'], 
                                           y_pred=self.merged['WAWR']) * 100
            self.result_dict['WAWR'] = {'Time (s)': time.time() - start, 
                                 'Golden Accuracy (%)': wawr_accuracy}
            print(f'{self.J} classes: Done WAWR')
            return self.predictions['WAWR'], worker_reliability
        elif self.J == 5:
            start = time.time()
            data_with_prob = self.toloka_estimates.merge(worker_reliability, left_on = 'worker_id', right_on = 'worker_id', how = 'left')
            data_with_prob = data_with_prob.fillna(0.5)
            res = data_with_prob.groupby(['task_id','label'])['reliability'].sum()
            res = res.unstack(level=1).fillna(0)
            res['probs'] = res.apply(lambda x : [x.values.tolist()], axis = 1)
            self.predictions['WAWR'] = res['probs'].apply(lambda x : 1 + np.argmax(x)).to_frame()
            test_data = self.predictions.merge(self.etoloka_estimates, on='task_id')
            result_i = self.predictions.merge(
                test_data.groupby('task_id')['label']
                .apply(lambda x: x.values.tolist())
                .to_frame(), 
                on='task_id'
                )
            result_i['pred'] = result_i.apply(lambda x: x['WAWR'] in x['label'], axis=1)
            self.result_dict['WAWR'] = {'Time (s)': time.time() - start, 
                                      'Golden Accuracy (%)': np.sum(result_i['pred']) / result_i['pred'].shape[0] * 100} 
            print(f'{self.J} classes: Done WAWR')
            return self.predictions['WAWR'], worker_reliability
    
    def GLAD(self, max_iter=2, bayes_likelyhood=False):
        '''
        Generative Model of Labels, Abilities and Difficulties (GLAD)
            link : http://papers.nips.cc/paper/3644-whose-vote-should-count
                -more-optimal-integration-
                of-labels-from-labelers-of-unknown-expertise.pdf

        Parameters
        ----------
        max_iter : int, optional
            max amount of EM iterations. The default is 2.
        bayes_likelyhood : bool, optional
            improve result by bayes_best_result. The default is False.

        Returns
        -------
        pd.Series: (self.row_shape, )
            prediction for all sites by GLAD.

        '''
        start = time.time()
        merged = self.toloka_estimates.merge(self.etoloka_estimates, on='task_id')
        self.glad = Glad(n_labels=2, 
                         n_workers=pd.unique(merged['worker_id']).shape[0], 
                         n_tasks=pd.unique(merged['task_id']).shape[0])
        res = self.glad.predict(merged, max_iter=max_iter)
        worker_reliability = self.glad.alpha
        worker_prob = pd.DataFrame(worker_reliability, columns=['reliability'], index=np.sort(pd.unique(merged['worker_id'])))
        worker_prob.index.name = 'worker_id'
        worker_prob['reliability'] = np.exp(worker_prob['reliability']) /  np.sum(np.exp(worker_prob['reliability']))
        data_with_prob = self.toloka_estimates.merge(worker_prob, left_on = 'worker_id', right_on = 'worker_id', how = 'left')
        data_with_prob = data_with_prob.fillna(0.5)
        if bayes_likelyhood:
            pr = self.bayes_best_result(start, data_with_prob, name='GLAD')  
            return pr, worker_reliability
        res = data_with_prob.groupby(['task_id','label'])['reliability'].sum()
        res = res.unstack(level=1).fillna(0)
        self.predictions['GLAD'] = res.apply(np.argmax, axis=1).astype(int)
        self.merged = self.predictions.merge(self.etoloka_estimates, on='task_id')
        wawa_accuracy = accuracy_score(y_true=self.merged['label'], 
                                       y_pred=self.merged['GLAD']) * 100
        self.result_dict['GLAD'] = {'Time (s)': time.time() - start, 
                             'Golden Accuracy (%)': wawa_accuracy}
        print(f'{self.J} classes: Done GLAD')
        return self.predictions['GLAD']
    