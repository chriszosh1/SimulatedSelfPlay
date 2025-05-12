from numpy import mean
import os

from Model import Model
from StageGames import BeautyContestGame, Symmetric2x2
from Agent import RandomAgent, CaseBasedAgent, SimpleReinforcementLearning_ER95
from Enums import Pairings, Burn_In, AggType
from Collect_Runs import collect_runs
from Data import session_aggregate

#def flatten_nested_list(nested_list):
#        '''Flattens the lists of data with arbitrary shape.'''
#        flattened = []
#        for item in nested_list:
#            if isinstance(item, list):
#               flattened.extend(flatten_nested_list(item))
#            else:
#                flattened.append(item)
#        return flattened

def fitness_mse(agg_data, agg_output, agg_type, var_weight = 0.05):
    mse = sum([(x - y) ** 2 for x, y in zip(agg_data[0], agg_output[0])])
    if agg_type == AggType.MEANANDWITHINVAR:
        mse += var_weight * sum([(x - y) ** 2 for x, y in zip(agg_data[1], agg_output[1])])
    mse = mse / 175 #rescaling by number of participants
    mse = mse / 66 #rescaling by the number of steps
    return mse

def get_model_fitness(data, get_agg_output, model_params, model_other_args, burn_in_type, runs, agg_type, delete_datafiles = False):  
    '''
    Evaluates the fitness of a given set of parameters for the model 
    Input:
        data - whatever source it may be
        get_agg_output - our aggregated model output generating function
        model_search_params - a dictionary of the set of parameters get_agg_output and bounds and grains we'll search over
        model_other_params - gives other params the model needs which we won't be searching over
        fitness_fnc - a toggle to decide which fitness type to use (mse or one including variance aswell)
    Output:
        bfp_fitness - the 'fitness score' of the chosen parameters
    '''
    fitness = 0
    for s in data:
        #1. Grabbing model args from session info:
        model_other_args["game_vars"]['game_params']['target_type'] = s['target_type']
        model_other_args['agent_count'] = s["n"]
        model_other_args['burn_in']['steps'] = s['steps']
        if model_other_args['burn_in']['burn_in_type'] == Burn_In.AGAINST_SELF:
            BI_tag = 'simselfplay'
        elif model_other_args['burn_in']['burn_in_type'] == Burn_In.AGAINST_OTHERS:
            BI_tag = 'classicburnin'
        else:
            BI_tag = 'nopreplay'
        model_other_args['output_vars'] = {'model_level_output': True, 'file_tag': f'{s["name"]}'}
        #1a. Updating model with model params searched over:
        if model_other_args['agent_vars']['agent_type'] == SimpleReinforcementLearning_ER95:
            model_other_args['agent_vars']['agent_params']['recency_bias'] = model_params['recency_bias']
            if 'attr_1' in model_params:
                model_other_args['agent_vars']['agent_params']['prior_strengths'] = [model_params['attr_1'], model_params['attr_2'], model_params['attr_3'], model_params['attr_4'], model_params['attr_5']]
            else:
                model_other_args['agent_vars']['agent_params']['prior_strengths'] = model_params['prior_strengths']
        if 'burn_in_rounds' in model_params:
            model_other_args['burn_in']['rounds'] = int(model_params['burn_in_rounds'])
        model_other_args['burn_in']['burn_in_type'] = burn_in_type
        

        #2. Run the aggregation function
        agg_output = get_agg_output(model_args = model_other_args, model_class = Model, runs = runs,
                                    agg_type = agg_type, steps = s['steps'], delete_datafiles = delete_datafiles)
        agg_data = s['agg_data']
        #3. Compute fitness
        #print(f'Agg Output: {agg_output}')
        #print(f'Agg Data: {agg_data}')
        fitness += fitness_mse(agg_data, agg_output, agg_type = agg_type)
            #3. Delete temp datafiles
    return fitness
