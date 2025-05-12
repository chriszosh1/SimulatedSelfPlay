from Model import Model
from StageGames import BeautyContestGame, Symmetric2x2
from Agent import RandomAgent, CaseBasedAgent, SimpleReinforcementLearning_ER95
from Enums import Pairings, Burn_In, BCGTargetType


burn_in_rounds = 1
steps = 100
agent_count = 10
output_vars = {'model_level_output': True, 'file_tag': 'model_sandbox_test_data'}

#-------Stage Game Specification Examples------------:
#game_vars = {'stage_game': BeautyContestGame, 'game_params': {'target_scalar': .5, 'prize': 20, 'max_choice':100, 'target_type':BCGTargetType.MEAN}} #BCG
game_vars = {'stage_game': Symmetric2x2, 'game_params': {'payoff_table': {0:{0:10, 1:30}, 1:{0:5, 1:25}}}} #the PD-CC09, where 0 is defect

#-------Agent Variable Specification Examples------------:
#agent_vars = {'agent_type': RandomAgent, 'agent_params': {}} #RANDOM AGENT
agent_vars = {'agent_type': SimpleReinforcementLearning_ER95, 'agent_params': {'prior_strengths':20, 'recency_bias':.05}} #RL95 Agent
#agent_vars = {'agent_type': CaseBasedAgent, 'agent_params': {'aspiration': 2, 'action_bandwidth': 10, 'sim_weight_action': 1,
#                                                             'state_space_params':{'period':{'weight':1},
#                                                                                   'target':{'weight':1, 'missing_sim':0},
#                                                                                   'burn_in':{'weight':1}}
#                                                             }} #CBDT for BCG
#agent_vars = {'agent_type': CaseBasedAgent, 'agent_params': {'aspiration': 2, 'action_bandwidth': 4, 'sim_weight_action': 1,
#                                                             'state_space_params':{'period':{'weight':1},
#                                                                                   'coop_rate':{'weight':1, 'missing_sim':0},
#                                                                                   'burn_in':{'weight':1}}
#                                                             }} #CBDT for Symmetric 2x2

#-------Burn In Specification Examples--------------------:
#burn_in = {}
burn_in = {'steps': steps, 'rounds': burn_in_rounds, 'burn_in_type':Burn_In.AGAINST_SELF}
#burn_in = {'steps': steps, 'rounds': burn_in_rounds, 'burn_in_type':Burn_In.AGAINST_SELF}

#-------Custom Pairing Specification Examples------------:
cp = [] #No custom - for RANDOM or N_PLAYER, optional so can also just not include
#cp = [[0,1,2,3],[4,5]] #CUSTOM_FIXED
#cp = [[[0,1],[2,3],[4,5]], [[0,1,2,3,4,5]], [[0,1],[2,3],[4,5]]] #CUSTOM_PER_PERIOD

test_model = Model(agent_count = agent_count, agent_vars = agent_vars,
                   game_vars = game_vars, output_vars = output_vars, verbose = True,
                   pairing_type=Pairings.N_PLAYER, custom_pairing=cp, burn_in = burn_in)
test_model.run_model(step_count = steps)