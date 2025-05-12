import numpy as np
from typing import Dict, Tuple, Callable

def behavioral_search(
    fitness_function: Callable,
    parameter_bounds: Dict[str, Tuple[float, float]],
    population_size: int = 50,
    max_iterations: int = 100,
    elite_size: int = 10,
    mutation_rate: float = 0.1,
    learning_rate: float = 0.3
) -> Tuple[float, Dict[str, float]]:
    """
    Performs behavioral search optimization to minimize the given fitness function.
    
    Args:
        fitness_function: Function that takes a dictionary of parameter names and values
                         and returns a float (lower is better)
        parameter_bounds: Dictionary mapping parameter names to (min, max) tuples
        population_size: Size of the population
        max_iterations: Maximum number of iterations
        elite_size: Number of best solutions to use for behavior learning
        mutation_rate: Probability of random mutation
        learning_rate: Rate at which to learn from elite behaviors
    
    Returns:
        Tuple of (best_fitness, best_parameters)
    """
    # Problem dimensionality
    n_dimensions = len(parameter_bounds)
    param_names = list(parameter_bounds.keys())
    
    # Convert bounds to arrays for easier processing
    bounds = np.array([parameter_bounds[param] for param in param_names])
    min_bounds = bounds[:, 0]
    max_bounds = bounds[:, 1]
    
    # Initialize population randomly
    population = np.random.uniform(
        min_bounds, max_bounds, 
        size=(population_size, n_dimensions)
    )
    
    # Initialize behavioral memory (mean and std of successful solutions)
    behavior_mean = np.mean(population, axis=0)
    behavior_std = np.std(population, axis=0)
    
    # Track best solution
    best_solution = None
    best_fitness = float('inf')
    
    # Main optimization loop
    for _ in range(max_iterations):
        print(f'Iteration {_}')
        # Evaluate fitness for all solutions
        fitness_values = []
        for sol in population:
            #print(f'Sol exploring: {sol}')
            args = dict(zip(param_names, sol))
            fitness_values.append(fitness_function(args))
        fitness_values = np.array(fitness_values)
        
        # Update best solution
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = population[current_best_idx].copy()
        
        # Select elite solutions
        elite_indices = np.argsort(fitness_values)[:elite_size]
        elite_solutions = population[elite_indices]
        
        # Learn from elite behaviors
        new_mean = np.mean(elite_solutions, axis=0)
        new_std = np.std(elite_solutions, axis=0)
        
        # Update behavioral memory with learning rate
        behavior_mean = (1 - learning_rate) * behavior_mean + learning_rate * new_mean
        behavior_std = (1 - learning_rate) * behavior_std + learning_rate * new_std
        
        # Generate new population based on learned behaviors
        population = np.random.normal(
            behavior_mean, 
            behavior_std, 
            size=(population_size, n_dimensions)
        )
        
        # Apply random mutations
        mutation_mask = np.random.random(population.shape) < mutation_rate
        random_values = np.random.uniform(min_bounds, max_bounds, population.shape)
        population[mutation_mask] = random_values[mutation_mask]
        
        # Enforce bounds
        population = np.clip(population, min_bounds, max_bounds)
        
        # Keep best solution
        population[0] = best_solution
        #print(f'Best solution this round: {population[0]}')
    
    # Convert best solution to dictionary
    best_parameters = dict(zip(param_names, best_solution))
    
    return best_fitness, best_parameters

#def test_fnc(x,y):
#    if x>2:
#        return x-y
#    else:
#        return x+y

#test = behavioral_search(fitness_function = test_fnc,
#                         parameter_bounds = {'x':(0,3), 'y':(1,4)},
#                         population_size = 50,
#                         max_iterations = 100,
#                         elite_size = 10,
#                         mutation_rate = 0.1,
#                         learning_rate = 0.3)
#print(test)