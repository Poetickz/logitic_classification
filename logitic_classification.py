#!/usr/bin/python3.5
""" test-linear-regression.py
    This script tests the Gradient Descent algorithm for multivariate
    linear regression.

    Author: Alan Rocha González
    Institution: Universidad de Monterrey
    First created: Mon 20 April, 2020
    Email: alan.rocha@udem.edu
"""
# import standard libraries
import numpy as np
import pandas as pd

# import user-defined libraries
import utilityfunctions as uf

def main():
  # runs main code
  """
  INPUT: NONE
  OUTPUT: NONE
  """
  # load training data
  x_training, y_training, x_testing, y_testing = uf.load_data('diabetes.csv')

  # declare and initialise hyperparameters
  learning_rate = 0.0005

  # Initailize all w
  w = np.array([[0.0]*x_training.T.shape[0]]).T
  
  # define stopping criteria
  stopping_criteria = 0.01

  # run the gradient descent method for parameter optimisation purposes
  w = uf.logistical(x_training, y_training, w, stopping_criteria, learning_rate)

  # evaluate with testing data and w
  uf.evaluate(w, x_testing, y_testing)

main()