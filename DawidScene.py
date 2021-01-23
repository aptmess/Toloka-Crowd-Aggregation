# -*- coding: utf-8 -*-
"""

@author: shiro
"""
import numpy as np
import pandas as pd


class DawidScene:
    
    def __init__(self, data, J=2, max_iter=10):
        '''
        Dawid Scene Implementation of Aggregation Crowdsoursed labels
           link : https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.469.1377

        Parameters
        ----------
        data : pd.DataFrame
            input crowdsoursed dataset.
        J : int, optional
            amount of classes. The default is 2.
        max_iter : int, optional
            amount of iteration in EM algorithm. The default is 10.

        Returns
        -------
        None.

        '''
        self.J = J
        self.max_iter = max_iter
        self.verdicts = np.arange(J)
        self.data = data.copy()
        self.preprocess()
        
    def preprocess(self, ):
        self.data.columns = ['worker_id', 'task_id', 'verdict']
        self.data = self.data.set_index('task_id')
        if self.J == 5:
            self.data['verdict'] = self.data['verdict'] - 1
        z_j = self.data.groupby(['task_id', 'verdict'])['worker_id'].count()
        z_j = z_j.unstack(level = 1).fillna(0)
        z_j = z_j.apply(lambda x : x / z_j.sum(axis = 1))
        z_j['z_j'] = z_j.apply(lambda x : [x.values.tolist()], axis = 1)
        self.z_j = z_j[['z_j']]
    
    def fit(self, trace=False, weights=0):
        '''
        EM iterations

        Parameters
        ----------
        trace : bool, optional
            print step by step information. The default is False.
        weights : int, optional
            choose type of worker_reliability calculation. The default is 1.
                - 1 : np.sum(np.diag(IEE)) / self.J
                - 0 : weighted : p * np.sum(np.diag(IEE)) / self.J
                        where p - is the distribution of classes

        Returns
        -------
        z_j_result : np.ndarray(amount_of_sites, self.J)
            probabilities of classes, latent variables of EM algorithm.
        e_w : np.ndarray(amount_of_tolokers, self.J, self.J)
            Individual Error Rates (self.J x self.J).
        p : np.ndarray(self.J, )
            distribution of classes.
        w_r : pd.DataFrame(amount_of_tolokers, )
            worker_reliability(weights).

        '''
        if trace:
            for k in range(self.max_iter):
                print(f'---------------START ITERATION #{k}-----------------')
                print('start M step')
                e_w, p = self.m_step(self.data, self.z_j)
                print('done M step')
                print('start E step')
                self.z_j = self.e_step(self.data, e_w, p)
                print('done E step')
                print(f'---------------END ITERATION #{k}-----------------')
            print('EM Done')
        else:
            for k in range(self.max_iter):
                e_w, p = self.m_step(self.data, self.z_j)
                self.z_j = self.e_step(self.data, e_w, p)
            print('EM Done')
        if self.J == 2:
            z_j_result = self.z_j['z_j'].apply(np.argmax).to_frame()
        else:
            z_j_result = self.z_j['z_j'].apply(lambda x : 1 + np.argmax(x)).to_frame()
        z_j_result = z_j_result.reset_index()
        z_j_result = z_j_result.sort_values(by='task_id')
        reabil = np.concatenate(e_w['e_w'].values, axis=0).reshape(-1, self.J, self.J)
        worker_reliab = {}
        if weights == 0:
            for i, num in zip(range(e_w.shape[0]), e_w.index):
                ie_rates = p.ravel() * reabil[i, :, :]
                reliability = np.sum(np.diag(ie_rates))
                worker_reliab[num] = reliability
        elif weights == 1:
            for i, num in zip(range(e_w.shape[0]), e_w.index):
                ie_rates = reabil[i, :, :]
                reliability = np.sum(np.diag(ie_rates))
                worker_reliab[num] = reliability * np.sum(p) / self.J
        w_r = pd.DataFrame(worker_reliab.values(), 
                           index=worker_reliab.keys(), 
                           columns=['reliability'])
        w_r.index.name = 'worker_id'
        return z_j_result, e_w, p, w_r
        
    def h1_m_step(self, i):
        l = np.zeros([len(self.verdicts), 1])
        l[i][0] = 1
        return l

    def h2_m_step(self, x):
        m = x.copy()
        l = np.sum(a = m, axis = 0)
        for i in range(len(m)):
            m[i] = np.divide(m[i], l, out = np.zeros_like(m[i]), where = l!=0)
        return m
    
    def m_step(self, data, z_j):
        '''
        Maximisation Step of EM Algorithm

        Parameters
        ----------
        data : pd.DataFrame
            input crowdsoursed dataset.
        z_j : np.ndarray(self.J, )
            probabilities of one class, latent variables of EM algorithm.

        Returns
        -------
        e_w : np.ndarray(amount_of_tolokers, self.J, self.J)
            Individual Error Rates (self.J x self.J).
        p : np.ndarray(self.J, )
            distribution of classes.

        '''
        e_w = data.merge(z_j, left_index = True, right_index = True, how = 'left')
        e_w['l'] = e_w['verdict'].apply(self.h1_m_step)
        e_w['e_w'] = e_w.apply(lambda x : np.matmul(x[3], x[2]), axis = 1)
        e_w = e_w.groupby('worker_id')['e_w'].apply(np.sum)
        e_w = e_w.apply(self.h2_m_step).to_frame()
        p = np.sum(z_j['z_j'].tolist(), axis = 0)
        p = p / np.sum(np.sum((p)))
        return e_w, p
    
    def h1_e_step(self, i):
        l = np.zeros([1, len(self.verdicts)])
        l[0][i] = 1
        return l
    
    def h2_e_step(self, x):
        m = x.copy()
        m = m / np.sum(m)
        return m
    
    def e_step(self, data, e_w, p):
        '''
        Expectation Step of EM algorithm

        Parameters
        ----------
        data : pd.DataFrame
            input crowdsoursed dataset.
        e_w : np.ndarray(amount_of_tolokers, self.J, self.J)
            Individual Error Rates (self.J x self.J).
        p : np.ndarray(self.J, )
            distribution of classes.

        Returns
        -------
        z_j : np.ndarray(self.J, )
            probabilities of one class, latent variables of EM algorithm.

        '''
        z_j = data.merge(e_w, left_on = 'worker_id', right_index = True, how = 'left')
        z_j['l'] = z_j['verdict'].apply(self.h1_e_step)
        z_j['z_j'] = z_j.apply(lambda x : np.matmul(x[3], x[2]), axis = 1)
        z_j = z_j.groupby('task_id')['z_j'].apply(np.prod).apply(lambda x : np.multiply(x, p))
        z_j = z_j.apply(self.h2_e_step).to_frame()
        return z_j
    
def list2array(J, inp):
    I, K, J = inp.shape[0], inp.shape[1], J
    print(f'''Количество данных: {I} \nКоличество толокеров: {K}\nКоличество классов: {J}''')
    dataset_tensor = np.zeros((I, K, J))
    for i in np.arange(I): 
        for k in np.arange(K):
            j = inp[i][k]
            dataset_tensor[i][k][j] += 1
    return dataset_tensor

class DawidSkeneModel:
    def __init__(self, J, names, max_iter = 100, tolerance = 0.01):
        self.J = J
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.names = names

    def run(self, dataset):
        self.I, self.K, _ = dataset.shape
        self.dataset_tensor = dataset
        T = self.dataset_tensor.sum(axis=1) / self.dataset_tensor.sum(axis=(1, 2)).reshape(-1, 1)
        flag = True
        prev_pi, prev_T = None, None
        iter_num = 0
        acc = {}
        while flag:
            print(f'Iteration Number: {iter_num + 1}')
            print('Start M-step')
            pi = self._m_step(T)
            print('Done M-step')
            print('Start E-step')
            next_T = self._e_step(T, pi)
            print('Done E-step')
            log_L = self._get_likelihood(T, pi)
            print(f'Done Get Likelyhood: {log_L}')
            if iter_num != 0:
                p_j = np.sum(T, 0) / self.I
                prev_p_j = np.sum(prev_T, 0) / self.I
                p_j_diff = np.sum(np.abs(p_j - prev_p_j))
                pi_diff = np.sum(np.abs(pi - prev_pi))

                if self._check_condition(p_j_diff, pi_diff, iter_num):
                    flag = False

            prev_pi = pi
            prev_T = T
            T = next_T
            iter_num += 1
            
            acc[iter_num] = {'LLH': log_L}
            print("-------------------------")

        worker_reliability = {}
        for i, num in zip(range(self.K), self.names):
            ie_rates = p_j * pi[i, :, :]
            reliability = np.sum(np.diag(ie_rates))
            worker_reliability[num] = reliability
            
        return p_j, pi, worker_reliability, T, acc

    def _check_condition(self, p_j_diff, pi_diff, iter_num):
        '''
        check where to end steps
        '''
        return (p_j_diff < self.tolerance and pi_diff < self.tolerance) or iter_num > self.max_iter

    def _m_step(self, T):
        '''
        M-Step: Maximisation LikelyHood
        '''
        pi = np.zeros((self.K, self.J, self.J))

        # Equation 2.3
        for i in range(self.J):
            pi_js = np.dot(T[:, i], self.dataset_tensor.transpose(1, 0 ,2))
            sum_pi_js = pi_js.sum(1)
            sum_pi_js = np.where(sum_pi_js == 0 , -10e9, sum_pi_js)
            pi[:, i, :] = pi_js / sum_pi_js.reshape(-1,1)                                                                        
        return pi
    
    def _e_step(self, T, pi):
        '''
        E-Step: Expectation Get
        '''
        p = T.sum(0) / self.I
        next_T = np.zeros([self.I, self.J])

        # Equation 2.5
        for i in range(self.I):
            class_likelood = self._get_class_likelood(pi, self.dataset_tensor[i])
            next_T[i] = p * class_likelood
            sum_p = next_T[i].sum()
            sum_p = np.where(sum_p == 0 , -10e9, sum_p)
            next_T[i] /= sum_p
        return next_T

    def _get_likelihood(self, T, pi):
        '''
        2.7 Equation: Likelyhood
        '''
        log_L = 0
        p = T.sum(0) / self.I

        # Equation 2.7
        for i in range(self.I):
            class_likelood = self._get_class_likelood(pi, self.dataset_tensor[i])
            log_L += np.log((p * class_likelood).sum())
        return log_L

    def _get_class_likelood(self, pi, task_tensor):
        # \sum_{j=1}^J p_{j} \prod_{k=1}^K \prod_{l=1}^J\left(\pi_{j l}^{(k)}\right)^{n_{il}^{(k)}}
        return np.power(pi.transpose(0, 2, 1), np.broadcast_to(task_tensor.reshape(self.K, self.J, 1), (self.K, self.J, self.J))).transpose(1, 2, 0).prod(0).prod(1)