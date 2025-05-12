import pandas as pd

from Enums import BCGTargetType, AggType


def session_aggregate(agg_type: AggType, lab_data: list) -> list:
    '''Generates our summary statistics to go into our fitness function'''
    ld = lab_data
    if agg_type == AggType.MEAN:
        results = [ld.mean().tolist()]
    elif agg_type == AggType.MEANANDWITHINVAR:
        results = [ld.mean().tolist(), ld.var().tolist()]
    return results

def grab_dn_data(agg_type: AggType) -> float: 
    """Imports Duffy Nagel data and specs about each session."""
    #Importing Data:
    sessions_data = {}
    for s_num in range(1,13):
        session_data = pd.read_excel(f'Data/DNSessions/Session{s_num}.xlsx', engine='openpyxl')
        sessions_data[s_num] = session_data

    #Assigning parameter values for each session for simulation:
    s_1 = {"name":"s_1", "target_type":BCGTargetType.MEDIAN, "n": 15, "steps":4, "lab_data":sessions_data[1], "agg_data": session_aggregate(agg_type, sessions_data[1])}
    s_2 = {"name":"s_2", "target_type":BCGTargetType.MEDIAN,"n": 15,"steps":4, "lab_data":sessions_data[2], "agg_data": session_aggregate(agg_type, sessions_data[2])}
    s_3 = {"name":"s_3", "target_type":BCGTargetType.MEDIAN,"n": 13,"steps":4, "lab_data":sessions_data[3], "agg_data": session_aggregate(agg_type, sessions_data[3])}
    s_4 = {"name":"s_4", "target_type":BCGTargetType.MEDIAN,"n": 13,"steps":10, "lab_data":sessions_data[4], "agg_data": session_aggregate(agg_type, sessions_data[4])}

    s_5 = {"name":"s_5", "target_type":BCGTargetType.MEAN,"n": 16,"steps":4, "lab_data":sessions_data[5], "agg_data": session_aggregate(agg_type, sessions_data[5])}
    s_6 = {"name":"s_6", "target_type":BCGTargetType.MEAN,"n": 14,"steps":4, "lab_data":sessions_data[6], "agg_data": session_aggregate(agg_type, sessions_data[6])}
    s_7 = {"name":"s_7", "target_type":BCGTargetType.MEAN,"n": 15,"steps":4, "lab_data":sessions_data[7], "agg_data": session_aggregate(agg_type, sessions_data[7])}
    s_8 = {"name":"s_8", "target_type":BCGTargetType.MEAN,"n": 14,"steps":10, "lab_data":sessions_data[8], "agg_data": session_aggregate(agg_type, sessions_data[8])}

    s_9 = {"name":"s_9", "target_type":BCGTargetType.MAX,"n": 15,"steps":4, "lab_data":sessions_data[9], "agg_data": session_aggregate(agg_type, sessions_data[9])}
    s_10 = {"name":"s_10", "target_type":BCGTargetType.MAX,"n": 15,"steps":4, "lab_data":sessions_data[10], "agg_data": session_aggregate(agg_type, sessions_data[10])}
    s_11 = {"name":"s_11", "target_type":BCGTargetType.MAX,"n": 15,"steps":4, "lab_data":sessions_data[11], "agg_data": session_aggregate(agg_type, sessions_data[11])}
    s_12 = {"name":"s_12", "target_type":BCGTargetType.MAX,"n": 15,"steps":10, "lab_data":sessions_data[12], "agg_data": session_aggregate(agg_type, sessions_data[12])}

    sessions = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9, s_10, s_11, s_12]
    return sessions