from random import shuffle
from copy import copy

from Enums import Pairings, Burn_In

class Model:
    '''A model to facilitate agents playing a repeated stage game.'''
    def __init__(self, agent_count: int,
                 pairing_type: Pairings,    #NOTE: If CUSTOM_FIXED, give list of lists of shape: custom_pairings = [[Group1 ids], [Group2 ids],...]
                                            #      If CUSTOM_BY_ROUND, give a list of the lists above, where each position is a round
                 agent_vars: dict,          #NOTE: agent_vars passes a dict of shape {'agent_type': class, agent_params: {}}
                 game_vars: dict,           #NOTE: game_vars passes a dict of shape {'stage_game': class, game_params: {}}
                 output_vars: dict,         #NOTE: output_vars passes a dict of shape {'model_level_output': bool, 'tag': str}
                 run: int = 0,              #For output storage purposes
                 verbose: bool = False,
                 custom_pairing: list = [], #A list of pairings is CUSTOM_FIXED, or a list of lists, where each sublist is the pairing each round if
                                            #CUSTOM_PER_PERIOD.
                 burn_in: dict = {}    #passes a dict of shape {'steps':steps, 'burn_in_type':Burn_In.enum}, determining how burn ins will occur if
                                            #at all. Passing in nothing or {} means no burn_in.
                 
                 ):
        #Model variables:
        self.period = 0
        self.run = run
        self.agent_count = agent_count
        self.pairing_type = pairing_type
        self.burn_in = burn_in
        if self.burn_in:
            self.burning_in = True
        else:
            self.burning_in = False  
        self.custom_pairings = custom_pairing
        self.period_pairing = None
        self.verbose = verbose
        #Setting up our stage game:
        self.game_params = game_vars.get('game_params')
        self.stage_game = game_vars.get('stage_game')(**self.game_params)
        #Making our agents:
        self.agent_params = agent_vars.get('agent_params')
        self.agents = {i : agent_vars.get('agent_type')(id = i, **self.agent_params) for i in range(agent_count)}
        self.alias_agents = {} #This will be used for self play in burn_in rounds in place of agents
        self.agent_ids = [i for i in range(self.agent_count)]
        #Giving agents lags (history) if any:
        initial_lags = self.return_states()
        for a in self.agents.values():
            a._store_lags(initial_lags)

        #Prepping storing our model output:
        self.output_vars = output_vars
        if self.output_vars.get('model_level_output', False):
            if self.run == 0:
                self.file = open(f'{self.output_vars.get("file_tag", "")}_playdata.txt','w')
                self.file.write('run,period,group_id,agent_id,choice,round_payoff\n')
                if self.burning_in:
                    self.file_burnin = open(f'{self.output_vars.get("file_tag", "")}_burnindata.txt','w')
                    self.file_burnin.write('run,period,group_id,agent_id,choice,round_payoff\n')
            else:
                self.file = open(f'{self.output_vars.get("file_tag", "")}_playdata.txt','a')
                if self.burning_in:
                    self.file_burnin = open(f'{self.output_vars.get("file_tag", "")}_burnindata.txt','a')
    
    def update_round_pairings(self) -> list:
        '''Establishes what the round pairings will be this round.'''
        if self.pairing_type == Pairings.N_PLAYER:
            return [self.agent_ids]
        elif self.pairing_type == Pairings.RANDOM:
            if self.agent_count % 2 == 0:
                #1. Mix up the order of all the ids
                random_order_ids = copy(self.agent_ids)
                shuffle(random_order_ids)
                #2. Chop up the shuffled id list into list of player pairs
                pairs = []
                for i in range(0, len(random_order_ids), 2):
                    pairs.append(random_order_ids[i:i+2])
                return pairs
            else:
                print('ERROR: An odd number of agents cannot be paired up!')
        elif self.pairing_type == Pairings.CUSTOM_FIXED:
            return self.custom_pairings
        elif self.pairing_type == Pairings.CUSTOM_BY_PERIOD:
            return self.custom_pairings[self.period]
        else:
            print('ERROR: Invalid pairing_type!')

    def collect_actions(self, stated_problem: dict, game_group_ids: list, agents: dict) -> dict:
        '''Asks the agents to make their choices and returns them.'''
        period_choices = {}
        for aid in game_group_ids:
            a = agents[aid]
            period_choices[aid] = a.choose(self.stage_game.return_action_set(), stated_problem)
        return period_choices
    
    def distribute_payoffs(self, payoffs: dict, outcome_info: dict, agents: dict, period_choices: dict) -> None:
        '''Gives payoffs and result info to the agents.'''
        #1. Get payoffs
        for aid in payoffs.keys():
            agents[aid].collect_payoff(payoffs.get(aid), self.burning_in, period_choices.get(aid))
        #2. Update attractions/memory
        for aid in payoffs.keys():
            agents[aid].update_mem(outcome_info)

    def store_output(self, id_list: list, group_id: int, period_choices: dict, payoffs: dict) -> None:
        '''Writes output to a text file.'''
        for aid in id_list:
            a = self.agents[aid]
            if self.burning_in:
                wf = self.file_burnin
            else:
                wf = self.file
            wf.write(f'{self.run},{self.period},{group_id},{aid},{period_choices[aid]},{payoffs[aid]}\n')

    def return_states(self):
        '''Returns period state info, adding model level info if needbe to the game info.'''
        ov = self.stage_game.return_outcome_info()
        ov['burn_in'] = int(self.burning_in)
        return ov
    
    def return_problem(self):
        '''Returns period problem, adding model level info if needbe to the game info.'''
        sp = self.stage_game.return_state_info(t=self.period)
        if self.burn_in:
            sp['burn_in'] = int(self.burning_in)
        return sp

    def _group_step(self, group_id, game_group_ids, agents = None):
        '''Steps through the model with just one of the groups.'''
        if agents is None:
            agents = self.agents
        #1. Present the problem to the agents:
        stated_problem = self.stage_game.return_state_info(t=self.period)
        #2. Ask agents to make a choice:
        period_choices = self.collect_actions(stated_problem, game_group_ids, agents)
        #3. Process the decisions to calculate payoffs:
        payoffs = self.stage_game.tabulate_game(period_choices)
        #4. Give agents their payoffs and outcome info:
        outcome_info = self.return_states()
        self.distribute_payoffs(payoffs, outcome_info, agents, period_choices)
        #5. Printing/Storing output:
        if self.output_vars.get('model_level_output', False):
            self.store_output(game_group_ids, group_id, period_choices, payoffs)
        if self.verbose:
            pc = {aid: period_choices[aid] for aid in agents.items()}
            rpo = {aid: payoffs[aid] for aid in agents.items()}
            print(f'period choices: {pc}')
            print(f'payoffs: {rpo}')
        #6. Agent Bookkeeping for next iteration:
        for a in set(agents.values()):
            a.bookkeeping()                      
        
    def create_alias(self, game_group_ids, self_play_id):
        '''Creates a temporary agent dict for the model to use when agents play themselves.'''
        alias_dict = {}
        for aid in game_group_ids:
            alias_dict[aid] = self.agents[self_play_id]
        return alias_dict

    def step(self):
        '''Asks the agents to play one iteration of the stage game.'''
        # Update pairings if necessary...
        self.period_pairing = self.update_round_pairings()
        if self.verbose:
            print(f'\n---Period {self.period}---')
            print(f'Pairings: {self.period_pairing}')
        #... then for each pairing/group:
        for group_id in range(len(self.period_pairing)):
            game_group_ids = self.period_pairing[group_id]
            #if self_play, each agent must simulate play themselves:
            if self.burning_in and self.burn_in['burn_in_type'] == Burn_In.AGAINST_SELF:
                for self_play_id in game_group_ids:
                    agent_aliases = self.create_alias(game_group_ids, self_play_id)
                    self._group_step(group_id, game_group_ids, agents = agent_aliases)
            #otherwise, play happens at the group level as per usual:
            else:
                self._group_step(group_id, game_group_ids)
            #Other Bookkeeping for next iteration:
            self.stage_game.bookkeeping()
            self.period += 1

    def run_model(self, step_count: int) -> None:
        '''Model steps steps number of times.'''
        if self.burning_in:
            if self.verbose:
                print('-BURNING IN-')
            for br in range(self.burn_in['rounds']):
                for bs in range(self.burn_in['steps']):
                    self.step()
                self.period = 0
            self.burning_in = False
            if self.verbose:
                print('\n-STEPPING-')
        #print('Post Burn-in Memories')
        #for ag in self.agents.values():
        #    print(f'\nAgent {ag.id}')
        #    for m in ag.memory:
        #        print(m)
        for s in range(step_count):
            self.step()
        #print('Post Game Memories')
        #for ag in self.agents.values():
        #    print(f'\nAgent {ag.id}')
        #    for m in ag.memory:
        #        print(m)
        self.file.close()
        self.file_burnin.close()
