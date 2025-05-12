import pandas as pd

from Data import grab_dn_data, session_aggregate
from Model import Model
from StageGames import BeautyContestGame, Symmetric2x2
from Agent import RandomAgent, CaseBasedAgent, SimpleReinforcementLearning_ER95
from Enums import Pairings, Burn_In, BCGTargetType, AggType
from Collect_Runs import collect_runs
from PS_Optimizer import bs_best_fit

#emptyHeaded = bs_best_fit(param_bounds = {'recency_bias':(0.01,0.99), 'prior_strengths':(0.01, 100)},
#                      search_rounds = 25, runs = 10, agg_type = AggType.MEANANDWITHINVAR,
#                      burn_in_type=Burn_In.AGAINST_OTHERS, filetag = 'emptyHeadedMV')
#print(emptyHeaded)

#classicBurnIn = bs_best_fit(param_bounds = {'burn_in_rounds':(0,5), 'recency_bias':(0.01,0.99), 'prior_strengths':(0.01, 100)},
#                      search_rounds = 25, runs = 10, agg_type = AggType.MEANANDWITHINVAR,
#                      burn_in_type=Burn_In.AGAINST_OTHERS, filetag = 'BurnInMV')
#print(classicBurnIn)

#simPlay = bs_best_fit(param_bounds = {'burn_in_rounds':(0,5), 'recency_bias':(0.01,0.99), 'prior_strengths':(0.01, 100)},
#                      search_rounds = 25, runs = 10, agg_type = AggType.MEANANDWITHINVAR,
#                      burn_in_type=Burn_In.AGAINST_SELF, filetag = 'SimPlayMV')
#print(simPlay)

FitBinPriors = bs_best_fit(param_bounds = {'recency_bias':(0.01,0.99), 'attr_1':(0.01, 100), 'attr_2':(0.01, 100), 'attr_3':(0.01, 100),
                                      'attr_4':(0.01, 100), 'attr_5':(0.01, 100)},
                      search_rounds = 25, runs = 10, agg_type = AggType.MEANANDWITHINVAR,
                      burn_in_type=Burn_In.AGAINST_OTHERS, filetag = 'FitBinPriors')
print(FitBinPriors)
