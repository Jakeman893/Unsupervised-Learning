import math

class Config:
    # Epsilon for stopping
    epsilon = 0.01


# TODO: Implement randomized hill climbing
def randomized_hill_climbing(current_value, acceleration = 1.2, bounds = (-1,1)):
    # Find five candidates for movement

    pass

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
    candidates[0] = -acceleration
    candidates[1] = -1/acceleration
    candidates[2] = 1 / acceleration
    candidates[3] = acceleration
    # Initialize best to be current score and positions
    best_score = eval_funct(current_position)
    best_position = current_position
    # Iterate through candidates getting max score
    for candidate in candidates:
        candidate = current_position + candidate * step_size
        score = eval_funct(candidate)
        if score > best_score:
            best_score = score
            best_position = candidate
    return best_position, best_score
