# -*- coding: utf-8 -*-
import numpy as np
import warnings
import pandas as pd
import sys
import scipy

warnings.simplefilter('ignore', RuntimeWarning)


class Glad:
    def __init__(self,n_labels, n_workers, n_tasks, nu=1.0):
        '''
        Generative Model of Labels, Abilities and Difficulties (GLAD)
            link : http://papers.nips.cc/paper/3644-whose-vote-should-count
                -more-optimal-integration-
                of-labels-from-labelers-of-unknown-expertise.pdf

        Parameters
        ----------
        n_labels : int
            amount of classes.
        n_workers : int
            amount of tolokers.
        n_tasks : int
            amount of sites.
        nu : float, optional
           scale parameter. The default is 1.0.
        self.alpha : np.ndarray(n_workers, )
            confidense of workers
        self.beta : np.ndarray(n_tasks, )
            difficulty of tasks
        self.prior_z: np.ndarray(n_labels, )
            prior step{0} probability of classes

        Returns
        -------
        None.

        '''
        self.n_labels = int(n_labels)
        self.n_workers = int(n_workers)
        self.n_tasks = int(n_tasks)
        self.nu = float(nu)
        self.alpha = np.random.normal(1, 1, n_workers)
        self.beta = np.exp(np.random.normal(1, 1, n_tasks))
        self.prior_z = np.ones(self.n_labels) / self.n_labels
        self.prior_z = np.array([0.4, 0.6])

    def get_status(self):
        print("   +-- < data status > --+")
        print("   |   n_labels  : {0:4d}  |".format(self.n_labels))
        print("   |   n_workers : {0:4d}  |".format(self.n_workers))
        print("   |   n_tasks   : {0:4d}  |".format(self.n_tasks))
        print("   +---------------------+")

    def get_ability(self):
        print("   +---- < Ability > ----+")
        for (i, a) in enumerate(self.alpha):
            print("   | No.{0:3d} :  {1:+.6f} |".format(i,a))
        print("   +---------------------+")

    def get_difficulty(self):
        print("   +--- < Difficulty > ---+")
        for (i, b) in enumerate(self.beta):
            print("   | No. {0:3d} :  {1:+.6f} |".format(i,b))
        print("   +----------------------+")

    def _mold_data(self, csv_data, expression=False):
        '''
        

        Parameters
        ----------
        csv_data : pd.DataFrame
            crowd input dataset.
        expression : bool, optional
            print information about steps. The default is False.

        Returns
        -------
        label_matrix : pd.DataFrame (self.n_workers, self.n_tasks)
            Labeled Preprocessed Matrix.

        '''
        if expression:
            print("Molding data")
        label_matrix = pd.pivot_table(csv_data, 
                                  index='task_id', 
                                      columns = 'worker_id', 
                                      values='label_x').values.T
        if expression:
            print("Finished to Mold data")
        return label_matrix
    
    def _log_sigma(self, x):
        '''
        Calculate Log Sigmoid

        Parameters
        ----------
        x : nd.ndarray
            inpur vector.

        Returns
        -------
        nd.ndarray
            LogSigmoid(x).

        '''
        return - np.maximum(0,-x)+np.log(1+np.exp(-np.abs(x)))

    def _Ilog_sigma(self, x):
        '''
        Calculate Log Sigmoid / K - 1

        Parameters
        ----------
        x : nd.ndarray
            inpur vector.

        Returns
        -------
        np.ndarray
            Calculate Log Sigmoid / K - 1

        '''
        return - np.log(self.n_labels - 1) - np.maximum(0,x)+np.log(1+np.exp(-np.abs(x)))

    def _E_step(self):
        '''
        Expectation Step

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        x = self.alpha[:, np.newaxis].dot(self.beta[np.newaxis, :])
        log_sigma = self._log_sigma(x)
        Ilog_sigma =  self._Ilog_sigma(x)

        def compute_likelihood(k):
            likelihood = np.where(self.label_matrix == k, log_sigma, Ilog_sigma)
            likelihood = np.where(np.isnan(self.label_matrix), 0, likelihood)
            return np.exp(likelihood.sum(axis = 0))

        post_z = np.array([compute_likelihood(i) * self.prior_z[i] for i in range(self.n_labels)])
        Z = post_z.sum(axis = 0)
        post_z = (post_z / Z).T
        if np.any(np.isnan(post_z)):
            sys.exit('Error:  Invalid Value [E_step]')
        return post_z

    def _Q_function(self, x):
        '''
        LogLikelyhood Function

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        new_alpha = x[:self.n_workers]
        new_beta = x[self.n_workers:]
        Q = (self.post_z * np.log(self.prior_z)).sum()
        x = new_alpha[:, np.newaxis].dot(new_beta[np.newaxis, :])
        log_sigma = self._log_sigma(x)
        Ilog_sigma =  self._Ilog_sigma(x)

        def compute_likelihood(k):
            log_likelihood = np.where(self.label_matrix == k, log_sigma, Ilog_sigma)
            log_likelihood = np.where(np.isnan(self.label_matrix), 0, log_likelihood)
            return log_likelihood

        z = np.array([compute_likelihood(i) for i in range(self.n_labels)])
        Q += (self.post_z * z.transpose((1,2,0))).sum()
        Q -= self.nu * ((new_alpha ** 2).sum() + (new_beta ** 2).sum())
        return Q

    def _MQ(self,x):
        return - self._Q_function(x)

    @staticmethod
    @np.vectorize
    def _sigma(x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    @np.vectorize
    def _Isigma(x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1.0 - 1e-15
        if x >= sigmoid_range:
            return 1e-15

        return 1.0 / (1.0 + np.exp(x))

    def _gradient(self, x):
        '''
        Calculate Gradient to Minimize Function
            depends on self.alpha, self.beta

        Parameters
        ----------
        x : np.ndarray
            input vector.

        Returns
        -------
        np.ndarray
            input vector.

        '''
        new_alpha = x[:self.n_workers]
        new_beta = x[self.n_workers:]
        y = new_alpha[:, np.newaxis].dot(new_beta[np.newaxis, :])
        sigma = scipy.special.expit(y)
        Isigma = scipy.special.expit(-y)

        def compute_likelihood(k):
            likelihood = np.where(self.label_matrix == k, Isigma, -sigma)
            likelihood = np.where(np.isnan(self.label_matrix), 0, likelihood)
            return likelihood

        z = np.array([compute_likelihood(i) for i in range(self.n_labels)])
        dQ_dalpha = (self.post_z * (z * new_beta).transpose((1,2,0))).sum(axis = (1,2)) - self.nu * new_alpha
        dQ_dbeta = (self.post_z * (z.transpose((0,2,1)) * new_alpha).transpose((2,1,0))).sum(axis = (0,2)) - self.nu * new_beta
        return np.r_[-dQ_dalpha, -dQ_dbeta]

    def _M_step(self, opt=False): 
        '''
        Maximisation Step in EM Algorithm

        Parameters
        ----------
        opt : bool, optional
            use ccipy.optimize.minimize function to calculate minimum of function. 
            if opt is True : use gradient descent. The default is False.

        Returns
        -------
        np.ndarray
            value of function.

        '''
        init_params = np.r_[self.alpha, self.beta]
        if opt:
            params = scipy.optimize.minimize(fun=self._MQ, 
                                       x0=init_params, 
                                       jac=self._gradient, 
                                       tol=0.01,
                                       options={'maxiter': 2, 
                                                'disp': False})
            self.alpha = params.x[:self.n_workers]
            self.beta = params.x[self.n_workers:]
            return -params.fun
        else: 
            for i in range(5):
                init_params -=  0.1 * self._gradient(init_params)
            print('calculate')
            self.alpha = init_params[: self.n_workers]
            self.beta = init_params[self.n_workers:]
            return -self._Q_function(init_params)

    def _EM_algo(self, tol, max_iter):
        '''
        Expectation Step of EM Algorithm

        Parameters
        ----------
        tol : float
            tolerance of differense on two steps.
        max_iter : int
            amount of iterations in EM algorithm.

        Returns
        -------
        np.ndarray(n_labels, self.J)
            probabilities of classes, latent variables of EM algorithm.

        '''
        self.post_z = self._E_step()
        alpha = self.alpha.copy()
        beta = self.beta.copy()
        x = np.r_[alpha, beta]
        now_Q = self._Q_function(x)
        for i in range(max_iter):
            prior_Q = now_Q
            print('start M step')
            now_Q = self._M_step()
            print('done M step')
            print('start E step')
            self.post_z = self._E_step()
            print('done M step')
            if np.abs((now_Q - prior_Q) / prior_Q) < tol:
                break
        return self.post_z

    def predict(self, data, tol=0.1, max_iter=1000):
        '''
        Start EM Algorithm with Preprocessing Data

        Parameters
        ----------
        data : pd.DataFrame
            crowd dataset data.
        tol : float, optional
            tolerance. The default is 0.1.
        max_iter : int, optional
            amount of iterations in EM Algorithm. The default is 1000.

        Returns
        -------
        np.ndarray(n_labels, self.J)
            probabilities of classes, latent variables of EM algorithm.

        '''
        self.label_matrix = self._mold_data(data, expression=True)
        self._EM_algo(tol, max_iter)
        return self.post_z