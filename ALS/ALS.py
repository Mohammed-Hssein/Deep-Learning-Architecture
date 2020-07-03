#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:39:03 2020

@author: mohammedhssein
"""
import numpy as np

class AlternatingLeastSquare():
    """
    
    """
    
    def __init__(self, n_iterations, n_factors, regularization):
        """
        

        Parameters
        ----------
        n_iterations : TYPE
            DESCRIPTION.
        n_factors : TYPE
            DESCRIPTION.
        regularization : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.n_iterations = n_iterations
        self.n_factors = n_factors
        self.regularization = regularization
        
    def fit(self, train, test):
        """
        

        Parameters
        ----------
        train : TYPE
            DESCRIPTION.
        test : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.n_user, self.n_items = train.shape
        self.user_factors = np.random.random(size=(self.n_user, self.n_factors))
        self.item_factors = np.random.random(size=(self.n_items, self.n_factors))
        
        
    
    def __alsAlgo(self, core, solve_vec, fixed_vec):
        """
        

        Parameters
        ----------
        core : Array 
            DESCRIPTION : ratings ... order 2 tensor containing the core values between X and Y.
            ------------
            Example : movieLens dataset X : users, Y:items, core ratings
            ---------    
        solve_vec : TYPE
            DESCRIPTION.
        fixed_vec : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass
    
    def predict(self):
        """
        

        Returns
        -------
        None.

        """
        pass
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """
        

        Parameters
        ----------
        y_true : TYPE
            DESCRIPTION.
        y_pre : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
