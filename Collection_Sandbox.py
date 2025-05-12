from Model import Model
from StageGames import BeautyContestGame, Symmetric2x2
from Agent import RandomAgent, CaseBasedAgent, SimpleReinforcementLearning_ER95
from Enums import Pairings, Burn_In, BCGTargetType, AggType
from Collect_Runs import collect_runs
from PlotFncs import plot_choices_over_time, plot_aggregated_choices


burn_in_rounds = 0
runs = 2
steps = 4
agent_count = 13

#-------Stage Game Specification Examples------------:
#gv = {'game_tag': 'non-strategic', 'stage_game': Symmetric2x2, 'game_params': {'payoff_table': {0:{0:1, 1:1}, 1:{0:1, 1:5}}}, 'action_range': (0,1)}    #g1_nonstrategic
#gv = {'game_tag': 'PD', 'stage_game': Symmetric2x2, 'game_params': {'payoff_table': {0:{0:2, 1:5}, 1:{0:1, 1:3}}}, 'action_range': (0,1), }   #g2_pd
gv = {'game_tag': 'BCG', 'stage_game': BeautyContestGame, 'game_params': {'target_scalar': .5, 'prize': 10, 'max_choice':100, 'target_type':BCGTargetType.MEAN}, 'action_range': (0,100)}    #g3_bcg

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

model_args = {'agent_count':agent_count, 'game_vars':gv, 'agent_vars': agent_vars,
              'output_vars': None, 'verbose': False, 'pairing_type':Pairings.N_PLAYER, 'custom_pairing':None,
              'burn_in': {'steps': steps, 'rounds': burn_in_rounds, 'burn_in_type':Burn_In.AGAINST_OTHERS} }

#output file tagging:
if model_args['burn_in']['burn_in_type'] == Burn_In.AGAINST_SELF:
    BI_tag = 'simselfplay'
elif model_args['burn_in']['burn_in_type'] == Burn_In.AGAINST_OTHERS:
    BI_tag = 'classicburnin'
else:
    BI_tag = 'nopreplay'
model_args['output_vars'] = {'model_level_output': True, 'file_tag': f'{gv["game_tag"]}_{BI_tag}_runs{runs}_periods{steps}_burnins{burn_in_rounds}'}

results =  collect_runs(Model, agg_type = AggType.MEANANDWITHINVAR, runs = runs, steps = steps, model_args = model_args)
print(results)
#plot_choices_over_time(f"{model_args['output_vars']['file_tag']}_playdata.txt", save_fig=True, output_path=f"{model_args['output_vars']['file_tag']}_play_choicesplot.png", yrange = gv['action_range'])
#plot_aggregated_choices(f"{model_args['output_vars']['file_tag']}_playdata.txt", save_fig=True, output_path=f"{model_args['output_vars']['file_tag']}_play_aggchoicesplot.png", yrange = gv['action_range'])