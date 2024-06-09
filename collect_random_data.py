import os
import numpy as np
from argparse import ArgumentParser
import csv

from utils.utils import *
import random_envs
import torch

from copy import deepcopy

import random

"""
	Examples:
		- python3 collect_random_data.py --env RandomHopper-v0 -n 100 [--output-dir hopper100]
		- python3 collect_random_data.py --env RandomHopper-v0 -n 100 --model policy.pth
		- python3 collect_random_data.py --env RandomHopper-v0 -n 100 --random-env --load-dr massonly_udr_test1.bounds --model policy.pth
		- python3 collect_random_data.py --env RandomHopper-v0 -n 100 --noise 1e-5
"""

def main(args):
	print('\nARGS:', vars(args))

	env = gym.make(args.env)

	set_seed(args.seed)

	sim_env_summary(env)

	if args.model != '':
		from utils.agent import Agent
		agent = Agent(env.observation_space.shape[0], env.action_space.shape[0],
		              replay_size=0, action_limit=args.action_limit, device="cpu")
		agent.policy_net.load_state_dict(torch.load(args.model, map_location="cpu"))

	if args.random_env:
		if not hasattr(env, 'load_dr_distribution_from_file'):
		    raise ValueError(f"Environment {args.env} does not have a load_dr_distribution_from_file method"
		                        "and args.random_env is True.")

		env.load_dr_distribution_from_file(args.load_dr)
		print('DR distribution loaded from file', args.load_dr)

	n_iter = args.n_iter

	states = []
	next_states = []
	actions = []
	terminals = []
	rewards = []

	state = env.reset()

	if args.random_env:
		env.set_random_task()

	# Generate random dataset of state transitions
	for i in range(n_iter):
		
		action = agent.get_action(state, sample=False).reshape(-1) if args.model != '' else np.random.randn((env.action_space.shape[0]))

		states.append(state)
		actions.append(action)
		
		state, reward, done, info = env.step(action)

		if args.noise > 0.0:
			next_states.append(state + np.sqrt(args.noise)*np.random.randn(state.shape[0]))
		else:
			next_states.append(state)

		if args.random_env and i%args.sample_interval == args.sample_interval-1:
			env.set_random_task()
			print('current environment:', env.get_task())

		if args.render:
			env.render()

		if done:
			terminals.append(True)
			state = env.reset()
		else:
			terminals.append(False)
		rewards.append(reward)

	# Shuffle all state transitions generated
	ds = []
	for state, next_state, action, terminal,reward in zip(states, next_states, actions, terminals,rewards):
		ds.append((state,next_state,action,terminal,reward))
	
	if args.shuffle:
		random.shuffle(ds)

	# ------------------ Formatting output array
	T = {'observations': None, 'next_observations': None, 'actions': None, 'terminals': None, 'rewards': None}

	T['observations'] = np.empty((0, len(states[0])), float)
	T['next_observations'] = np.empty((0, len(states[0])), float)
	T['actions'] = np.empty((0, len(actions[0])), float)
	T['terminals'] = np.empty((0,), bool)
	T['rewards'] = np.empty((0,),float)

	for state, next_state, action, terminal,reward in (ds):
		T['observations'] = np.append(T['observations'], np.array([state]), axis=0)
		T['next_observations'] = np.append(T['next_observations'], np.array([next_state]), axis=0)
		T['actions'] = np.append(T['actions'], np.array([action]), axis=0)
		T['terminals'] = np.append(T['terminals'], np.array([terminal]), axis=0)
		T['rewards'] = np.append(T['rewards'],np.array([reward]),axis=0)
	
	T['terminals'][-1] = True

	
	# # ----------- Reproduce them just to make sure

	# Reinitialize env with target default dynamics
	env = gym.make(args.env)

	transitions = [i for i in range(n_iter-1)]

	mse = []
	env.reset()
	template = deepcopy(env.get_sim_state())

	for t in transitions:
		state = T['observations'][t]
		# next_state = T['observations'][t+1]
		next_state = T['next_observations'][t]
		action = T['actions'][t]
		terminal = T['terminals'][t]
		reward = T['rewards'][t]

		if terminal == True:
			print('Terminal state:', t)
			print('It has not been counted in the MSE\n ------')
			continue

		env.set_sim_state(env.get_full_mjstate(state, template))

		ob_prime, r, d, _ = env.step(action)

		mse.append((np.linalg.norm(ob_prime-next_state)**2))

	print('Mean:', np.array(mse).mean())

	if args.output_dir != '':
		os.makedirs('customdatasets/'+str(args.output_dir))

		np.save('customdatasets/'+str(args.output_dir)+'/'+str(args.env)+'_n'+str(args.n_iter)+'_observations.npy', T['observations'])
		np.save('customdatasets/'+str(args.output_dir)+'/'+str(args.env)+'_n'+str(args.n_iter)+'_nextobservations.npy', T['next_observations'])
		np.save('customdatasets/'+str(args.output_dir)+'/'+str(args.env)+'_n'+str(args.n_iter)+'_actions.npy', T['actions'])
		np.save('customdatasets/'+str(args.output_dir)+'/'+str(args.env)+'_n'+str(args.n_iter)+'_terminals.npy', T['terminals'])
		np.save('customdatasets/'+str(args.output_dir)+'/'+str(args.env)+'_n'+str(args.n_iter)+'_rewards.npy', T['rewards'])


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("--env", type=str, default='RandomHopper-v0', help="Environment")
	parser.add_argument("--render", "-r", default=False, action='store_true', help="Render")
	parser.add_argument("--seed", type=int, default=0, help="Numpy seed")
	parser.add_argument("--n_iter", "-n", type=int, default=800000, help="Number of iterations")
	parser.add_argument("--model", type=str, default="RandomHopper-v0_policy_15.pth", help="model for exploring agent - random data is collected if not specified (default)")
	parser.add_argument("--action-limit", type=float, default=1., help="Maximum action value")
	parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
	parser.add_argument("--noise", type=float, default=0.0, help="Variance of additive noise to each observation dimension (default: 0.0)")
	parser.add_argument("--random-env", default=True, action='store_true', help="Whether to sample new dynamics after each episode")
	parser.add_argument("--load-dr", type=str, default="dr.csv", help="File name containing the bounds for the distribution. Only if --random_env is set. (Default: '')")
	parser.add_argument("--sample-interval", type=int, default=1000, help="sample new dynamics every --sample-interval transitions")
	parser.add_argument("--shuffle", "-s", default=False, action='store_true', help="Shuffle collected data")
	parser.add_argument("--output-dir", type=str, default="uni_model_800k", help="Output directory for the dataset created (default: '' => no saving")
	args = parser.parse_args(); print(args)
	

	main(args)