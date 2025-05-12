from enum import Enum

class Pairings(Enum):
    N_PLAYER = 0
    RANDOM = 1
    CUSTOM_FIXED = 2
    CUSTOM_BY_PERIOD = 3

class Burn_In(Enum):
    AGAINST_OTHERS = 0
    AGAINST_SELF = 1

class BCGTargetType(Enum):
    MEAN = 0
    MEDIAN = 1
    MAX = 2

class AggType(Enum):
    MEAN = 0
    MEANANDWITHINVAR = 1