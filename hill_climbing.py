__author__ = "Jacob Logas"

# Some code templates courtesy of http://www.cleveralgorithms.com/nature-inspired/stochastic/hill_climbing_search.html

import math
import numpy as np

"""
Returns a random bitstring
"""
def random_bitstring(num_bits):
    return np.random.randint(0,2,num_bits)

"""
Returns the score of a bitstring with goal of all 1's (The sum of the bitstring)
"""
def score_bitstring(bs):
    return np.sum(bs)

"""
Input:
    Standard Hill Climbing
    eval_funt: lambda that takes a X as input and returns a score
    move_funct: function that retrieves a move given X and k
    k: A function that is able to be used as an "acceleration" for move function
    X: value indicating where the current configuration
    epsilon: the minimum difference between a previous score and the new score
Returns:
    X: the location of maximal score found
    X_score: the maximal score found
"""
def hill_climbing(eval_funct, move_funct, X, k=1, epsilon=0.01):
    X_score = eval_funct(X)
    prev_score = float("-inf")
    while not np.isclose(prev_score, X_score, epsilon):
        prev_score = X_score
        prev_X = X
        candidates = [None] * 5
        candidates[0] = -k
        candidates[1] = -1 / k
        candidates[2] = 0
        candidates[3] = 1 / k
        candidates[4] = k

        E_i = [None] * 5

        for i in xrange(0,5):
            E_i[i] = eval_funct(move_funct(X, candidates[i]))

        max_idx = np.argmax(E_i)

        X = move_funct(X, candidates[max_idx])
        X_score = E_i[max_idx]
        
        # If we are getting the same output decelerate
        if prev_score == X_score and prev_X == X:
            k /= 2
        # print(X,X_score)
    return X, X_score

"""
Randomized Hill Climbing
Input:
    eval_funt: lambda that takes a position as input and returns a score
    start: value indicating where the algorithm will start
    epsilon: the minimum difference between a previous score and the new score
    step_size: The maximum amount the algorithm can move per iteration
    acceleration: Modifies the step size for how much to move
    bounds: The window of the function to consider, defaults to no limit
    prob_jump: The probability that on any iteration the function will choose a new value
    iter_lim: The total iteration limit for the algorithm to run
Returns:
    current: the value with current max score
    current_score: the maximal score found
"""
# Implement randomized hill climbing
def randomized_hill_climbing(eval_funct, move_funct, X, k=1, prob_jump=0.01, iter_lim=1000, epsilon=0.01):
    X_score = eval_funct(X)
    prev_score = float("-inf")
    iter_cnt = 0
    while not np.isclose(prev_score, X_score, epsilon) and iter_cnt < iter_lim:
        iter_cnt += 1
        # Jump with probability given to random point in figure
        if np.random.rand() < prob_jump:
            X = move_funct(X, np.random.uniform(-100, 100))
            X_score = eval_funct(X)

        # Then just regular hill climbing
        prev_score = X_score
        candidates = [None] * 5
        candidates[0] = -k
        candidates[1] = -1 / k
        candidates[2] = 0
        candidates[3] = 1 / k
        candidates[4] = k

        E_i = [None] * 5

        for i in xrange(0,5):
            E_i[i] = eval_funct(move_funct(X, candidates[i]))

        max_idx = np.argmax(E_i)

        X = move_funct(X, candidates[max_idx])
        X_score = E_i[max_idx]
        # print (X,X_score)
   
    return X, X_score


"""
Simulated Annealing
Input:
    eval_funt: lambda that takes a position as input and returns a score
    start: value indicating where the algorithm will start
    T: The starting temperature
    min_T: When to stop annealing
    T_decay: The rate of decay of temperature
    iters_per_decay: Number of annealing iterations to perform per temperature decay
    bounds: The window of the function to consider, defaults to no limit
Returns:
    current: the location of maximal score found
    current_score: the maximal score found
"""
# TODOA: Implement simulated annealing
def simulated_annealing(eval_funct, candidate_funct, start, T = 1.0, min_T = 0.0001, T_decay=0.9, iters_per_decay=100, bounds = (float("-inf"),float("inf"))):
    current = start
    current_score = eval_funct(start)
    while T > min_T:
        for _ in xrange(0,iters_per_decay):
            # Choose new random position within bounds
            new_pos = candidate_funct(bounds)
            new_score = eval_funct(new_pos)
            if new_score > current_score \
            or math.exp((new_score - current_score)/T) > np.random.rand():
                current = new_pos
                current_score = new_score
        T = T * T_decay
    return current, current_score

"""
Hill Climber Iterative Value Move
This is a movement function that gets values from a range
X: The current position on the x axis
i: Basically the acceleration
step_size: The largest step in either direction to go
bounds: The window in which the values can come from
"""
def maximum_number_move(X, i, step_size=1, bounds=(float("-inf"), float("inf"))):
    return np.clip(X + (i * step_size), bounds[0], bounds[1])