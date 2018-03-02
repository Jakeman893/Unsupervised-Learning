__author__ = "Jacob Logas"

import math
import numpy as np

class Config:
    # Epsilon for stopping
    epsilon = 0.01

"""
Input:
    Standard Hill Climbing
    eval_funt: lambda that takes a position as input and returns a score
    start_position: value indicating where the algorithm will start
    epsilon: the minimum difference between a previous score and the new score
    step_size: The maximum amount the algorithm can move per iteration
    acceleration: Modifies the step size for how much to move
    bounds: The window of the function to consider, defaults to no limit
Returns:
    current_pos: the location of maximal score found
    current_score: the maximal score found
"""
def hill_climbing(eval_funct, candidate_funct, start_position, epsilon=0.01, step_size=1, acceleration=1.2, bounds=(float("-inf"),float("inf"))):
    current_pos = start_position
    current_score = eval_funct(start_position)
    prev_score = float("-inf")
    while not np.isclose(prev_score, current_score, epsilon):
        prev_score = current_score
        prev_pos = current_pos
        current_pos, current_score = hill_climbing_step(eval_funct, candidate_funct, current_pos, step_size, acceleration, bounds)
        # If we are getting the same output decelerate
        if prev_score == current_score and prev_pos == current_pos:
            acceleration /= 2
    return current_pos, current_score

"""
Randomized Hill Climbing
Input:
    eval_funt: lambda that takes a position as input and returns a score
    start_position: value indicating where the algorithm will start
    epsilon: the minimum difference between a previous score and the new score
    step_size: The maximum amount the algorithm can move per iteration
    acceleration: Modifies the step size for how much to move
    bounds: The window of the function to consider, defaults to no limit
    prob_jump: The probability that on any iteration the function will choose a new value
    iter_lim: The total iteration limit for the algorithm to run
Returns:
    current_pos: the location of maximal score found
    current_score: the maximal score found
"""
# Implement randomized hill climbing
def randomized_hill_climbing(eval_funct, candidate_funct, start_position, prob_jump=0.01, iter_lim=1000, epsilon=0.01, step_size=1, acceleration=1.2, bounds=(float("-inf"),float("inf"))):
    current_pos = start_position
    current_score = eval_funct(start_position)
    prev_score = float("-inf")
    iter_cnt = 0
    while not np.isclose(prev_score, current_score, epsilon) and iter_cnt < iter_lim:
        iter_cnt += 1
        # Jump with probability given to random point in figure
        if np.random.rand() < prob_jump:
            current_pos = candidate_funct(bounds)
            current_score = eval_funct(current_pos)
        prev_score = current_score
        prev_pos = current_pos
        current_pos, current_score = hill_climbing_step(eval_funct, candidate_funct, current_pos, step_size, acceleration, bounds)
        # If we are getting the same output decelerate
        if prev_score == current_score and prev_pos == current_pos:
            acceleration /= 2
    
    return current_pos, current_score

"""
Simulated Annealing
Input:
    eval_funt: lambda that takes a position as input and returns a score
    start_position: value indicating where the algorithm will start
    T: The starting temperature
    min_T: When to stop annealing
    T_decay: The rate of decay of temperature
    iters_per_decay: Number of annealing iterations to perform per temperature decay
    bounds: The window of the function to consider, defaults to no limit
Returns:
    current_pos: the location of maximal score found
    current_score: the maximal score found
"""
# TODO: Implement simulated annealing
def simulated_annealing(eval_funct, candidate_funct, start_position, T = 1.0, min_T = 0.0001, T_decay=0.9, iters_per_decay=100, bounds = (float("-inf"),float("inf"))):
    current_pos = start_position
    current_score = eval_funct(start_position)
    while T > min_T:
        for i in xrange(0,iters_per_decay):
            # Choose new random position within bounds
            new_pos = candidate_funct(bounds)
            new_score = eval_funct(new_pos)
            if new_score > current_score \
            or math.exp((new_score - current_score)/T) > np.random.rand():
                current_pos = new_pos
                current_score = new_score
        T = T * T_decay
    return current_pos, current_score

# TODO: Implement genetic algorithm
def genetic(fit_funct, candidate_funct, bit_vect):
    # TODO: Generate initial guess

    # TODO: Iterate while not equal or while iterations are less than max
        # TODO: Get fitness of guess
        # TODO: Mutate
    pass

# TODO: Implement MIMIC
def MIMIC():
    pass

# TODO: Implement fitness function
def fitness():
    pass

# Performs a single step of the hill climbing algorithm
#  Using a passed in eval function, iteratively looks at neighboring values
#  and finds the best scorer
#  Returns: best_position and best_score
def hill_climbing_step(eval_funct, candidate_funct, current_position, step_size=1, acceleration = 1.2, bounds = (-1, 1)):
    # Find five candidates for movement
    candidates = [None] * 4
    # Different candidate steps to take
    candidates[0] = current_position + -acceleration * step_size
    candidates[1] = current_position + -1/acceleration * step_size
    candidates[2] = current_position + 1 / acceleration * step_size
    candidates[3] = current_position + acceleration * step_size
    # Clip candidates to the window provided
    candidates = np.clip(candidates, bounds[0], bounds[1])
    # Initialize best to be current score and positions
    best_score = eval_funct(current_position)
    best_position = current_position
    # Iterate through candidates getting max score
    for candidate in candidates:
        score = eval_funct(candidate)
        if score > best_score:
            best_score = score
            best_position = candidate
        # If equal flip coin and go either left or right
        elif score == best_score:
            if np.random.rand() < 0.5:
                best_score = score
                best_position = candidate
    # print (best_position, best_score)
    return best_position, best_score
