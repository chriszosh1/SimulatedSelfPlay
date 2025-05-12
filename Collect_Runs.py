import pandas as pd
import numpy as np
import os

from Model import Model
from Enums import AggType


def collect_runs(model_class, agg_type, runs, steps, model_args, delete_datafiles = True):
    '''Creates models runs times and collects data from each.'''
    #1. Running each model:
    for r in range(runs):
        temp_model = model_class(run = r, **model_args)
        temp_model.run_model(step_count = steps)
    #2. Opening up the data file and importing it to be evaluated:
    filename = f"{model_args['output_vars']['file_tag']}_playdata.txt"
    #print(filename)
    dataframe = pd.read_csv(filename, sep=',')
    #print(dataframe)
    if agg_type == AggType.MEAN:
        df = dataframe.groupby('period').agg({'choice': 'mean'}).reset_index()
        stats = [df['choice'].tolist()]
    elif agg_type == AggType.MEANANDWITHINVAR:
        df = dataframe.groupby('period').agg({'choice': ['mean', 'var']}).reset_index()
        stats = df['choice']['mean'].tolist(), df['choice']['var'].tolist()
    if delete_datafiles:
        os.remove(filename)
        os.remove(f"{model_args['output_vars']['file_tag']}_burnindata.txt")    
    return stats