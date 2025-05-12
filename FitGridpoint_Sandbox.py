import pandas as pd

from Data import grab_dn_data, session_aggregate
from Model import Model
from StageGames import BeautyContestGame, Symmetric2x2
from Agent import RandomAgent, CaseBasedAgent, SimpleReinforcementLearning_ER95
from Enums import Pairings, Burn_In, BCGTargetType, AggType
from Collect_Runs import collect_runs
from Fitness import get_model_fitness, fitness_mse
from PlotFncs import plot_choices_over_time


#Test run:
data = grab_dn_data(agg_type=AggType.MEANANDWITHINVAR) #Shape: {"name":"s_1", "target_type":"median", "n": 15, "steps":4, "lab_data":sessions_data[1]}
evaldata = data[1:3] + [data[4]] + [data[11]] #Sessions 1,4,6-11 used for training. 2,3,5,and 12 are used for eval.

runs = 25

#-------Stage Game Specification Examples------------:
#gv = {'game_tag': 'non-strategic', 'stage_game': Symmetric2x2, 'game_params': {'payoff_table': {0:{0:1, 1:1}, 1:{0:1, 1:5}}}, 'action_range': (0,1)}    #g1_nonstrategic
#gv = {'game_tag': 'PD', 'stage_game': Symmetric2x2, 'game_params': {'payoff_table': {0:{0:2, 1:5}, 1:{0:1, 1:3}}}, 'action_range': (0,1), }   #g2_pd
gv = {'game_tag': 'BCG', 'stage_game': BeautyContestGame, 'game_params': {'target_scalar': .5, 'prize': 10, 'max_choice':100, 'target_type':None}, 'action_range': (0,100),}    #g3_bcg

#-------Agent Variable Specification Examples------------:
#agent_vars = {'agent_type': RandomAgent, 'agent_params': {}} #RANDOM AGENT
agent_vars = {'agent_type': SimpleReinforcementLearning_ER95, 'agent_params': {'prior_strengths':10, 'recency_bias':0.05}} #RL95 Agent
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
burn_in_rounds = 0
model_args = {'agent_count':None, 'game_vars':gv, 'agent_vars': agent_vars,
              'output_vars': None, 'verbose': False, 'pairing_type':Pairings.N_PLAYER, 'custom_pairing':None,
              'burn_in': {'steps': None, 'rounds': burn_in_rounds, 'burn_in_type':Burn_In.AGAINST_OTHERS} }
#Note: Nones are grabbed fromt he sessions themselves

result = get_model_fitness(data = evaldata, get_agg_output = collect_runs, burn_in_type = Burn_In.AGAINST_SELF, 
                           model_params = {'burn_in_rounds':4, 'prior_strengths':68.068, 'recency_bias':0.732},
                         model_other_args=model_args, runs = runs, agg_type = AggType.MEANANDWITHINVAR,
                         delete_datafiles = False)

file = open(f'EvalFitBinPriorsMV.txt','w')
file.write(f'recency_bias,attr_1,attr_2,attr_3,attr_4,attr_5,fitness\n')
file.write(f'0.822,0.01,55.468,51.603,10.794,5.254,{result}')
print(result)
plot_choices_over_time('s_12_playdata.txt', save_fig=True, output_path='test_choices_plot.png', yrange = (0,100))