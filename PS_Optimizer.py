import pandas as pd
from typing import Dict

from Data import grab_dn_data, session_aggregate
from Model import Model
from StageGames import BeautyContestGame, Symmetric2x2
from Agent import RandomAgent, CaseBasedAgent, SimpleReinforcementLearning_ER95
from Enums import Pairings, Burn_In, BCGTargetType, AggType
from Collect_Runs import collect_runs
from Fitness import get_model_fitness, fitness_mse
from Optimize_BS import behavioral_search


#Test run:
data = grab_dn_data(agg_type=AggType.MEANANDWITHINVAR) #Shape: {"name":"s_1", "target_type":"median", "n": 15, "steps":4, "lab_data":sessions_data[1]}
training_data = [data[0]] + [data[3]] + data[5:11] 
#-------Stage Game Specification Examples------------:
#gv = {'game_tag': 'non-strategic', 'stage_game': Symmetric2x2, 'game_params': {'payoff_table': {0:{0:1, 1:1}, 1:{0:1, 1:5}}}, 'action_range': (0,1)}    #g1_nonstrategic
#gv = {'game_tag': 'PD', 'stage_game': Symmetric2x2, 'game_params': {'payoff_table': {0:{0:2, 1:5}, 1:{0:1, 1:3}}}, 'action_range': (0,1), }   #g2_pd
gv = {'game_tag': 'BCG', 'stage_game': BeautyContestGame, 'game_params': {'target_scalar': .5, 'prize': 10, 'max_choice':100, 'target_type':None}, 'action_range': (0,100),}    #g3_bcg

#-------Agent Variable Specification Examples------------:
#agent_vars = {'agent_type': RandomAgent, 'agent_params': {}} #RANDOM AGENT
agent_vars = {'agent_type': SimpleReinforcementLearning_ER95, 'agent_params': {'prior_strengths':None, 'recency_bias':None}} #RL95 Agent
#agent_vars = {'agent_type': CaseBasedAgent, 'agent_params': {'aspiration': 2, 'action_bandwidth': 4, 'sim_weight_action': 1,
#                                                             'state_space_params':{'period':{'weight':1},
#                                                                                   'target':{'weight':1, 'missing_sim':0}}
#                                                             }} #CBDT for BCG
#agent_vars = {'agent_type': CaseBasedAgent, 'agent_params': {'aspiration': 2, 'action_bandwidth': 4, 'sim_weight_action': 1,
#                                                             'state_space_params':{'period':{'weight':1},
#                                                                                   'coop_rate':{'weight':1, 'missing_sim':0}}
#                                                             }} #CBDT for Symmetric 2x2

#-------Custom Pairing Specification Examples------------:
#cp = [] #No custom - for RANDOM or N_PLAYER
#cp = [[0,1,2,3],[4,5]] #CUSTOM_FIXED
#cp = [[[0,1],[2,3],[4,5]], [[0,1,2,3,4,5]], [[0,1]]] #CUSTOM_PER_PERIOD

model_args = {'agent_count':None, 'game_vars':gv, 'agent_vars': agent_vars,
              'output_vars': None, 'verbose': False, 'pairing_type':Pairings.N_PLAYER, 'custom_pairing':None,
              'burn_in': {'steps': None, 'rounds': 0, 'burn_in_type':None} }
#Note: Nones are grabbed fromt he sessions themselves

def bs_best_fit(param_bounds, burn_in_type, search_rounds = 100, runs = 25, agg_type = AggType.MEANANDWITHINVAR,filetag = ''):
    def ff_helper(params: Dict):
        '''Reformats our fitness function to be amenable with our search function'''
        results =  get_model_fitness(data = training_data, get_agg_output = collect_runs, model_params = params,
                                    model_other_args=model_args, burn_in_type = burn_in_type, runs = runs, agg_type = agg_type, delete_datafiles=True)
        return results
    final_results = behavioral_search(fitness_function = ff_helper,
                         parameter_bounds = param_bounds,
                         population_size = 25,
                         max_iterations = search_rounds,
                         elite_size = 5,
                         mutation_rate = 0.1,
                         learning_rate = 0.3)
    file = open(f'Training{filetag}_bestfit.txt','w')
    param_names = ",".join(f"{k}" for k in param_bounds.keys())
    file.write(f'{param_names},fitness\n')
    param_vals = ",".join(f'{final_results[1][k]}' for k in param_bounds.keys())
    file.write(f'{param_vals},{final_results[0]}')
    return final_results