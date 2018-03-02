import math
import numpy as np

class Config:
    # Epsilon for stopping
    epsilon = 0.01

def hill_climbing(eval_funct, start_position, epsilon=0.01, step_size=1, acceleration=1.2, bounds=(float("-inf"),float("inf"))):
    current_pos = start_position
    current_score = eval_funct(start_position)
    prev_score = float("-inf")
    while not np.isclose(prev_score, current_score, epsilon):
        prev_score = current_score
        prev_pos = current_pos
        current_pos, current_score = hill_climbing_step(eval_funct, current_pos, step_size, acceleration, bounds)
        # If we are getting the same output decelerate
        if prev_score == current_score and prev_pos == current_pos:
            acceleration /= 2
    return current_pos, current_score

# TODO: Implement randomized hill climbing
def randomized_hill_climbing(eval_funct, start_position, prob_jump=0.01, iter_lim=1000, epsilon=0.01, step_size=1, acceleration=1.2, bounds=(float("-inf"),float("inf"))):
    current_pos = start_position
    current_score = eval_funct(start_position)
    prev_score = float("-inf")
    iter_cnt = 0
    while not np.isclose(prev_score, current_score, epsilon) and iter_cnt < iter_lim:
        iter_cnt += 1
        # Jump with probability given to random point in figure
        if np.random.rand() < prob_jump:
            current_pos = np.random.uniform(bounds[0], bounds[1])
            current_score = eval_funct(current_pos)
        prev_score = current_score
        prev_pos = current_pos
        current_pos, current_score = hill_climbing_step(eval_funct, current_pos, step_size, acceleration, bounds)
        # If we are getting the same output decelerate
        if prev_score == current_score and prev_pos == current_pos:
            acceleration /= 2
    
    # At end of iterations, check if hill climber was finished
    if iter_cnt == iter_lim:
        return hill_climbing(eval_funct, current_pos, epsilon, step_size, acceleration, bounds)

    return current_pos, current_score

# TODO: Implement simulated annealing
def simulated_annealing():
    pass

# TODO: Implement genetic algorithm
def genetic():
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
def hill_climbing_step(eval_funct, current_position, step_size=1, acceleration = 1.2, bounds = (-1, 1)):
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
