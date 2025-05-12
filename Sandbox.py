action_set = [j for j in range(0,101)]
S = [1,2,3,4,5,6,7,8,9,10]

subsize = len(S)
action_set_size = len(action_set)
base_size = action_set_size // subsize
remainder = action_set_size % subsize
i = 0
s_counter = 0
action_attractions = {}
for act in action_set:
    action_attractions[act] = S[s_counter]
    i += 1
    if s_counter == 0 and i == base_size + remainder:
        s_counter+=1
        i = 0
    elif s_counter > 0 and i == base_size:
        s_counter+=1
        i = 0

print(action_attractions)