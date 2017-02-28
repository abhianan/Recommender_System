# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:20:17 2016

@author: Abhishek
"""

from numpy import *
num_news=10
num_users=5

ratings = random.randint(11, size = (num_news, num_users))
did_rate = (ratings != 0) * 1

abhi_ratings = zeros((num_news, 1))

abhi_ratings[0] = 8
abhi_ratings[4] = 7
abhi_ratings[7] = 3

ratings = append(abhi_ratings, ratings, axis = 1)
did_rate = append(((abhi_ratings != 0) * 1), did_rate, axis = 1)

#-----------------------------------------------------------------------------

def normalize_ratings(ratings, did_rate):
    num_news = ratings.shape[0]
    
    ratings_mean = zeros(shape = (num_news, 1))
    ratings_norm = zeros(shape = ratings.shape)
    
    for i in range(num_news): 
        idx = where(did_rate[i] == 1)[0]
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    return ratings_norm, ratings_mean
    
ratings, ratings_mean = normalize_ratings(ratings, did_rate)
num_users = ratings.shape[1]
num_features = 10

news_features = random.randn( num_news, num_features )
user_prefs = random.randn( num_users, num_features )

initial_X_and_theta = r_[news_features.T.flatten(), user_prefs.T.flatten()]

#--------------------------------------------------------------------------------

def unroll_params(X_and_theta, num_users, num_news, num_features):
	first_30 = X_and_theta[:num_news * num_features]
	X = first_30.reshape((num_features, num_news)).transpose()
	last_18 = X_and_theta[num_news * num_features:]
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta
 
#--------------------------------------------------------------------------------
 
def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_news, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_news, num_features)
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

#--------------------------------------------------------------------------------

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_news, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_news, num_features)
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2 
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2)) 
	return cost + regularization
 
#-----------------------------------------------------------------------------------
from scipy import optimize
reg_param = 30

minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta,  args=(ratings, did_rate, num_users, num_news, num_features, reg_param), maxiter=100, disp=True, full_output=True ) 
cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]
movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_news, num_features)
#-------------------------------------------------------------------------------------------------------------
all_predictions = news_features.dot( user_prefs.T )+ratings_mean

predictions_for_abhi = all_predictions[:, 0:1]

print predictions_for_abhi