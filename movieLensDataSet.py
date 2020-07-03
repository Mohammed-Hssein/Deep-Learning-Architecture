#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:12:24 2020

@author: mohammed

Please find datasets at : https://grouplens.org/datasets/movielens/
"""
# DATA PREPROCESSING
import numpy as np
import pandas as pd
import torch


import sys
sys.path.append('./ressources/')
import RBM_V0 as RBM

# IMportingthe dataset

""" MOvie ID, Name of movie, Type of movie """
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine= 'python', 
                     encoding= 'latin-1')

""" User ID , Sex, age, some income classes, """
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine= 'python', 
                     encoding= 'latin-1')

""" User ID, movie ID, Ratings from 1 to 5, Time stamps when users rated movie """
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine= 'python', 
                     encoding= 'latin-1')

# Preparing the trainig set and the test set

training_set = pd.read_csv('ml-100k/u1.base', delimiter= '\t') # delimiter is tab
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter= '\t') # delimiter is tab
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies

nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into matrix where users are in rows and movies in columns

def  convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set) # LIst of nb_users size in wich each element is rating of movies
test_set = convert(test_set)

# convert to torch tensor

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Convertingthe data into liked or not liked 

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


nv = len(training_set[0])
nh = 100
batch_size = 70 # update the weights after batch_size observations

nb_epochs = 20
steps = 10


rbm = RBM.RestrictetBoltzmanMachine(nv,nh)


rbm.train(training_set, steps, batch_size, nb_epochs, loss="mean")
rbm.test(test_set)
