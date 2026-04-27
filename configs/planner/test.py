import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

walk_state_traj_bool = [False,True,True,True,True, False, False,False,False,True, True, True]
transitions_down = (walk_state_traj_bool[:-1] == True) & (walk_state_traj_bool[1:] == False)
transitions_up = (walk_state_traj_bool[:-1] == False) & (walk_state_traj_bool[1:] == True)

indices_down = np.where(transitions_down)[0]
indices_up = np.where(transitions_up)[0]
print(f"The indecies:{indices_up} {indices_down}")

walk_state_traj_soft = np.zeros(len(walk_state_traj_bool))
if len(walk_state_traj_bool) > 0 and walk_state_traj_bool[0]:
    walk_state_traj_soft = np.ones(len(walk_state_traj_bool))
import pdb; pdb.set_trace()
for i, indice in enumerate(indices_down):
    walk_state_traj_soft += np.array([sigmoid(t-indice) for t in range(len(walk_state_traj_bool))])
for i, indice in enumerate(indices_up):
    walk_state_traj_soft -= np.array([sigmoid(t-indice) for t in range(len(walk_state_traj_bool))])

print(f"The walk state soft: {walk_state_traj_soft}")

walk_state_traj_bool_np = np.array(walk_state_traj_bool, dtype=bool)
transitions_down = (walk_state_traj_bool_np[:-1] == True) & (walk_state_traj_bool_np[1:] == False)
transitions_up = (walk_state_traj_bool_np[:-1] == False) & (walk_state_traj_bool_np[1:] == True)

indices_down = np.where(transitions_down)[0]
indices_up = np.where(transitions_up)[0]


walk_state_traj_soft = np.zeros(len(walk_state_traj_bool))

if len(walk_state_traj_bool_np) > 0 and walk_state_traj_bool_np[0]:
    walk_state_traj_soft = np.ones(len(walk_state_traj_bool))
pdb.set_trace()
for i, indice in enumerate(indices_down):

    walk_state_traj_soft -= np.array([sigmoid(t-indice) for t in range(len(walk_state_traj_bool))])

for i, indice in enumerate(indices_up):

    walk_state_traj_soft += np.array([sigmoid(t-indice) for t in range(len(walk_state_traj_bool))])

print(f"The walk state soft: {walk_state_traj_soft}")