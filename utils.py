import numpy as np
from copy import deepcopy
import gym


def set_seed(seed):
	if seed > 0:
		np.random.seed(seed)


# Hopper-env-specific
def get_full_mjstate(state, template):
    # Get a new fresh mjstate template
    mjstate = deepcopy(template)

    mjstate.qpos[0] = 0.
    mjstate.qpos[1:] = state[:5]
    mjstate.qvel[:] = state[5:]

    return mjstate


def sim_env_summary(envsim):
	print('\n ---- Environment summary -----')
	print('Qpos dim:', envsim.model.nq)
	print('Qvel dim (DOFs):', envsim.model.nv)
	print('Ctrl dim (# of actuators):', envsim.model.nu)
	print('\n')
	print('DOFs per body:', [(body_name, dofs) for body_name, dofs in zip(envsim.model.body_names, envsim.model.body_dofnum)])
	print('Qpos and Qvel indexes per joint:', [(joint_name, envsim.model.get_joint_qpos_addr(joint_name), envsim.model.get_joint_qvel_addr(joint_name)) for joint_name in envsim.model.joint_names])
	print('Actuated joints:', [envsim.model.joint_id2name(id) for id in envsim.model._actuator_name2id.values()])

