""" utililtyfunctions.py
    This script have all utility fuctions

    Author: Alan Rocha González
    Institution: Universidad de Monterrey
    First created: Mon 20 April, 2020
    Email: alan.rocha@udem.edu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys


def eval_hypothesis_function(w, x):
    """
    INPUT: w: numpy array with all w values
           x: numpy array with all x data set values
    OUTPUT: Return the multiplication of the w.T & x.T
    """
    """Evaluate the hypothesis function"""
    return 1/(1+np.exp(np.matmul(-w.T,x.T)))


def compute_gradient_of_cost_function(x, y, w):
    """compute gradient of cost function"""
    """
    INPUT: w: numpy array with all w values
           x: numpy array with all x data set values
           y: numpy array with all y data set values
    OUTPUT: Return the gradient_of_cost_function
    """
    # compute number of training samples
    Nfeatures = x.shape[1]
    Nsamples = x.shape[0]


    # evaluate hypotesis function
    hypothesis_function = eval_hypothesis_function(w, x)

    # compute difference between hypothesis function and label
    y = y.T
    residual =  hypothesis_function - y

    # multiply the residual by the input features x; 
    gradient_of_cost_function = np.matmul(residual,x)

    # sum the result up and divide the total by the number of samples N
    gradient_of_cost_function = sum(gradient_of_cost_function)/Nsamples
    
    # reshape the gradient of cost function from a 1xNsample to a Nsamplex1 matrix
    gradient_of_cost_function = np.reshape(gradient_of_cost_function,(Nfeatures,1))

    # return the gradient of cost function
    return gradient_of_cost_function


def compute_L2_norm(gradient_of_cost_function):
    """compute L2-norm of gradient of cost function"""
    """
    INPUT: gradient_of_cost_function
    OUTPUT: Return the sum of all square element of gradient_of_cost_function
    """
    return np.linalg.norm(gradient_of_cost_function)

def feature_scaling(data, mean_value, std_value):
    """ standarize the x data and saves mean value & std value"""
    """
    INPUT: data: data from de data set that will be standarized (numpy array)
           mean_value: mean_value (float)
           std_value: standard variation value (float)
    OUTPUT: Returns de data set standarized, the mean value and std value
    """
    if mean_value == 0 and std_value == 0:
        std_value=data.std()
        mean_value=data.mean()
    scaling_scores = (data - mean_value) / std_value
    return scaling_scores, mean_value, std_value

def shuffler(filename):
    """ shuffle data of the csv file """
    """
    INPUT: filename: the csv file name
    OUTPUT: Return the shuffled dataframe
    """
    df = pd.read_csv(filename, header=0)
    # return the pandas dataframe
    return df.reindex(np.random.permutation(df.index))

def load_data(path_and_filename):
    """ load data from comma-separated-value (CSV) file """
    """
    INPUT: path_and_filename: the csv file name
    OUTPUT: Return the x_training data (numpy array), y_training data(numpy array), 
            x_testing data (numpy array float) and y_testing (numpy array float)
    """

    # Opens file
    try:
        training_data = shuffler(path_and_filename)

    except IOError:
      print ("Error: El archivo no existe")
      exit(0)

    # Gets rows and columns
    n_rows, n_columns = training_data.shape

    # Gets the testing set
    testing_x = pd.DataFrame.to_numpy(training_data.iloc[:int(n_rows*.20),0:n_columns-1])
    testing_y = pd.DataFrame.to_numpy(training_data.iloc[:int(n_rows*.20),-1]).reshape(int(n_rows*.20),1)

    # Gets the training set
    x_data = pd.DataFrame.to_numpy(training_data.iloc[int(n_rows*.20):,0:n_columns-1])
    y_data = pd.DataFrame.to_numpy(training_data.iloc[int(n_rows*.20):,-1]).reshape((n_rows-int(n_rows*.20)),1)

    # Prints Training Data
    print("\n")
    print("--"*23)
    print("Training Data")
    print("--"*23)
    for x,y in zip(x_data, y_data):
        print(x, y)

    # Prints Testing Data"
    print("\n")
    print("--"*23)
    print("Testing Data")
    print("--"*23)
    for x,y in zip(testing_x, testing_y):
        print(x, y)

    # Initializate lists
    mean = []
    std = []
    training_x = []
    new_testing_x = []

    # Applying feature scaling for training set
    for feature in x_data.T:
        dataScaled, meanX, stdX = feature_scaling(feature, 0, 0)
        training_x.append(np.array(dataScaled))
        mean.append(meanX)
        std.append(stdX)

    training_x = np.array(training_x).T



    # Applying feature scaling for testing set
    for feature,mean,std in zip (testing_x.T,mean,std):
        dataScaled = feature_scaling(feature,mean,std)
        new_testing_x.append(np.array(dataScaled[0]))

    testing_x = np.array(new_testing_x).T


    # create a ones-vector of size as x for training set

    one_data = np.ones(shape=(len(training_x.T[0])))

    # create a ones-vector of size as x for testing set
    one_data_2 = np.ones(shape=(len(testing_x.T[0])))

    # concatenate one_data with training_x
    training_x = np.column_stack((one_data,training_x))
    # concatenate one_data with testing_x
    testing_x = np.column_stack((one_data_2,testing_x))


    # Prints Training Data Scaled"
    print("\n")
    print("--"*23)
    print("Training Data Scaled")
    print("--"*23)
    for x,y in zip(training_x, y_data):
        print(x, y)

    # Prints Testing Data Scaled
    print("\n")
    print("--"*23)
    print("Testing Data Scaled")
    print("--"*23)
    for x,y in zip(testing_x, testing_y):
        print(x, y)
    print("\n\n\n")


    return training_x, y_data, testing_x, testing_y

def logistical(x_training, y_training, w, stopping_criteria, learning_rate):
    """ run the gradient descent algorith for optimisation"""
    """
    INPUT: w: numpy array with all w values
           x_training: numpy array with all x data set values
           y_training: numpy array with all y data set values
           stopping_criteria: float value
           learning rate: float value
    OUTPUT: Returns the w values in a numpy array float
    """
    # gradient descent algorithm

    # Initiate progress bar variables
    counter = 0.01
    limit = stopping_criteria/counter
    progress = 0

    print("Computing function...\n")
    while True:

        # compute gradient of cost function
        gradient_of_cost_function = compute_gradient_of_cost_function(x_training,
                                                                      y_training,
                                                                      w)
        # update parameters
        w = w - learning_rate*gradient_of_cost_function

        # compute L2 Norm
        L2_norm = compute_L2_norm(gradient_of_cost_function)
        if L2_norm < stopping_criteria:
            break

        # Progress bar
        if L2_norm <= limit:
            counter += 0.01
            limit = stopping_criteria/counter
            if (round((counter*100),2) % 5 == 0):
                progress += 1
            sys.stdout.write("{}[{}] {}".format(str("\b"*30),
                                                str(("="*progress)+(" "*(20-progress))),
                                                ("%"+str(round((counter*100),2)))))
            sys.stdout.flush()
    # Print w parameters
    print("\nDone!")
    print("\n")
    print("--"*23)
    print("w parameter")
    print("--"*23)
    for i in range(0,len(w)):
        print("w"+str(i)+": "+str(w[i][0]))
    return w
def print_perfomance_metrics(tp,tn, fp, fn):
    """ Display confusion matrix and performance metrics"""
    """
    INPUT:  tp: True positive (count)
            tn = True negative (count)
            fp = False positive (count)
            fn = False negative (count)
    OUTPUT: NONE
    """

    #Prints confusion matrix variables
    print("TP: "+str(tp))
    print("TN: "+str(tn))
    print("FP: "+str(fp))
    print("FN: "+str(fn))

    # Calculate accuracy
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    # Calculate precision
    precision = (tp)/(tp+fp)
    # Calculate recall
    recall = (tp/(tp+fn))
    # Calculate specifity
    specifity = (tn/(tn+fp))
    # Calculate f1 score
    f1 = (2.0*((precision*recall)/(precision+recall)))

    # Print performance metrics
    print("Accuracy:"+str(accuracy))
    print("Precision:"+str(precision))
    print("Recall:"+str(recall))
    print("Specifity:"+str(specifity))
    print("F1 Score: " + str(f1))

def evaluate(w, x_testing, y_testing):
    """ Evaluate testing set"""
    """
    INPUT:  w: the numpy array of w values
            x_testing: the numpy array of x values
            y_testing: the numpy array of y values
    OUTPUT: NONE
    """
    print("-----------------------------------------")

    # Initiate variables to the confusion matrix
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Evaluate x_testing
    for x,y in zip(x_testing,y_testing):
        predict = np.matmul(w.T,x.T)
        if(predict > 0 and y == 1):
            tp += 1
        if(predict < 0 and y == 0):
            tn += 1
        if(predict < 0 and y == 1):
            fn += 1
        if(predict > 0 and y == 0):
            fp += 1

    # Prints Confusion Matrix and some metrics
    print_perfomance_metrics(tp,tn, fp, fn)
