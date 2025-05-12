from random import choice, choices
from numpy import exp
from math import sqrt
from copy import copy
from typing import Union


class _Agent:
    '''Agent base class'''
    def __init__(self, id: int):
        self.id = id
        self.round_payoff = 0
        self.period_choice = None
        self.lags = {}        
    
    def _update_benefit(self, payoff: float) -> None:
        '''Updates basic payoff variables'''
        self.round_payoff = payoff

    def _store_lags(self, outcome_info:dict) -> None:
        '''Saves this periods info as lags for next period.'''
        self.lags = copy(outcome_info)

    def update_mem(self, outcome_info: dict):
        '''Updates memory or attractions if needbe'''
        pass

    def bookkeeping(self):
        '''Prepares for the next round.'''
        pass

class RandomAgent(_Agent):      #agent_params takes the following form.. {}
    '''Agent chooses from available actions uniform randomly.'''
    def __init__(self, id: int):
        super().__init__(id)

    def choose(self, action_set, stated_problem) -> int:
        '''Given actions available and state info, returns choice from action set.'''
        self.period_choice = choice(action_set)
        return self.period_choice

    def collect_payoff(self, payoff: float, outcome_info: dict, period_choice: float) -> None:
        '''Agent take their payoff and update their choice performance forecasts.'''
        self._update_benefit(payoff)

class SimpleReinforcementLearning_ER95(_Agent):
    '''Agent class learns through reinforcement on actions, ignoring state, as seen in Erev&Roth95.'''
    def __init__(self, id: int, prior_strengths: Union[float, dict, list], recency_bias: float, verbose: bool = False):
        super().__init__(id)
        self.R = recency_bias
        self.S = prior_strengths    #NOTE: Can give float for uniform priors, or give dict of shape {act1:float, act2:float...}
        self.action_attractions = {} # Dictionary to store attractions for each action
        self.action_probabilities = {}
        self.verbose = verbose

    def initialize_attractions(self, action_set: list):
        if isinstance(self.S, int) or isinstance(self.S, float):
            self.action_attractions = {action: self.S for action in action_set}
        elif isinstance(self.S, dict):
            self.action_attractions = {act:self.S[act] for act in action_set}
        elif isinstance(self.S, list):
            subsize = len(self.S)
            action_set_size = len(action_set)
            base_size = action_set_size // subsize
            remainder = action_set_size % subsize
            i = 0
            s_counter = 0
            for act in action_set:
                self.action_attractions[act] = self.S[s_counter]
                i += 0
                if s_counter == 0 and i == base_size + remainder:
                    s_counter+=1
                    i = 0
                elif s_counter > 0 and i == base_size:
                    s_counter+=1
                    i = 0

    def choose(self, action_set: list, stated_problem: dict) -> int:
        '''Given actions available and state info, returns choice from action set.'''
        #1. Create attraction vector if doesn't exist yet:
        if not self.action_attractions:
            self.initialize_attractions(action_set)
        
        #2. Choose by drawing action randomly, proportional to attractions:
        attractions = list(self.action_attractions.values())
        total = sum(attractions)
        weights = [attract / total for attract in attractions]
        self.period_choice = choices(action_set, weights=weights, k=1)[0]
        
        #3. Save and return choice:
        self.period_stated_problem = stated_problem
        #print(f'{self.id} chose {self.period_choice}')
        return self.period_choice

    def collect_payoff(self, payoff: float, outcome_info: dict, period_choice: float) -> None:
        if self.verbose:
              print(f"Before updating, attractions for agent {self.id}: {self.action_attractions}")
        self._update_benefit(payoff)
        for action in self.action_attractions.keys():
            self.action_attractions[action] *= (1 - self.R)
            if action == period_choice:
                self.action_attractions[action] += payoff 
        if self.verbose:
            print(f"After updating, attractions for agent {self.id}: {self.action_attractions}")


class CaseBasedAgent(_Agent):   #agent_params takes the following form.. {'aspiration': _, 'action_bandwidth':_, sim_weight_action:_,'state_space_params':{'var1':{'weight':_, 'missing':_}, 'var1':{'weight':_, 'missing':_}, ...}
    '''Agent case based reasoning class with bandwidth action sim.'''
    def __init__(self, id: int, aspiration: float, sim_weight_action: float, action_bandwidth: int, state_space_params: dict, lags_as_state:bool = True):
        super().__init__(id)
        self.aspiration = aspiration
        self.action_bandwidth = action_bandwidth
        self.sim_weight_action = sim_weight_action
        self.state_space_params = state_space_params
        self.lags_as_state = lags_as_state
        self.round_attractions_updated = False
        self.memory = []        #a list of cases (dictionaries), of the following form.. [{'circumstance': {}, 'action':_, 'result':_}, ...]
        self.action_attractions = {}

    def _reset_attractions(self, action_set: list) -> None:
        '''Establishes and resets attractions for new calcs each period.'''
        for a in action_set:
            self.action_attractions[a] = 0

    def _establish_problem(self, stated_problem: dict) -> None:
        if self.lags_as_state:
            self.period_stated_problem = {**stated_problem, **self.lags}
        else:
            self.period_stated_problem = stated_problem

    def _update_attractions(self, case: dict, action_set: list) -> None:
        '''Returns vector of action_attractions case will contribute.'''
        #1. Establish range of actions which whose estimates will be affected by this experience (case):
        min_affected = max(case.get('action') - self.action_bandwidth, min(action_set))
        max_affected = min(case.get('action') + self.action_bandwidth, max(action_set))
        #2. Calculate part of similarity contributed by dimensions of the stated problem:
        sim_tally = 0
        for problem_dimension, state_val in self.period_stated_problem.items():
            #print(f'PERIOD PROBLEM: {self.period_stated_problem}')
            if problem_dimension in self.state_space_params:
                #print(f'problem_dimension: {problem_dimension}')
                #print(f'state_val: {state_val}')
                #print(f"case: {case}")
                #print(f"dimension in case: {case['circumstance'][problem_dimension]}")
                #print(f'state_val? {(state_val)}')
                if (state_val is not None) and (case['circumstance'][problem_dimension] is not None):
                    sim_tally += self.state_space_params[problem_dimension]['weight']*((case['circumstance'][problem_dimension] - state_val)**2)
                else:
                    sim_tally += self.state_space_params[problem_dimension]['missing_sim']
            else:
                print(f'ERROR - Missing dimension of stated problem in agent state space params: {problem_dimension}')
        for act in action_set:
            if act >= min_affected and act <= max_affected:
                #3. Calculate part of similarity contributed by the action similarity:
                sim_action = self.sim_weight_action*((case['action'] - act)**2)
                sim_score = exp(-sqrt(sim_tally + sim_action))
                #4. Calculate CBU for that action and add it into our action attractions
                self.action_attractions[act] += sim_score*(case.get('result') - self.aspiration)
        self.round_attractions_updated = True

    def choose(self, action_set: list, stated_problem: dict) -> int:
        '''Given actions available and state info, returns choice from action set.'''
        #0. Establishes the problem the agent is facing:
        self._establish_problem(stated_problem)
        #1. Reset attractions to prepare for new calcs:
        self._reset_attractions(action_set)
        #2. Update action estimates by processing each case in memory:
        if not self.round_attractions_updated:
            for case in self.memory:
                self._update_attractions(case, action_set)
        #3. Make choice:
        max_cbu = max(self.action_attractions.values())
        period_choice = choice([act for act, cbu in self.action_attractions.items() if cbu == max_cbu])
        #4. Save and return choice:
        self.period_choices.append(period_choice)
        return period_choice

    def collect_payoff(self, payoff: float, outcome_info: dict, period_choice: float) -> None:
        '''Agent take their payoff and update their choice performance forecasts.'''      
        #Updates basic payoff variables
        self._update_benefit(payoff)

    def update_mem(self, outcome_info: dict):
        '''Updates memory or attractions if needbe'''
        #Save outcome info as lags to reference later:
        self._store_lags(outcome_info)
        #Stores experience to memory
        #print(f'self.period_statd_problem: {self.period_stated_problem}')
        #print(f'self.period_choices: {self.period_choices}')
        #print(f'self.round_payoff: {self.round_payoff}')
        for p in range(len(self.period_choices)):
            self.memory.append({'circumstance': copy(self.period_stated_problem), 'action': self.period_choices[p], 'result': self.round_payoff[p]})

    def bookkeeping(self):
        '''Prepares for the next round.'''
        self.round_attractions_updated = False
        self.period_choices = None
        self.round_payoff = 0