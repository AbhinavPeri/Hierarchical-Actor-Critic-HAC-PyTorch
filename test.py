import torch
import gymnasium as gym
import gymnasium_robotics
import asset
import numpy as np
from HAC import HAC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
    #################### Hyperparameters ####################
    env_name = "PointMaze-h-v3"
    save_episode = 10               # keep saving every n episodes
    max_episodes = 1000             # max num of training episodes
    random_seed = 0
    render = False
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    action_high = env.action_space.high
    action_low = env.action_space.low
    action_bounds = (action_high - action_low) / 2
    action_bounds = torch.FloatTensor(action_bounds.reshape(1, -1)).to(device)
    action_offset = (action_high + action_low) / 2
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = action_low # np.array([-1.0 * action_bounds])
    action_clip_high = action_high # np.array([action_bounds])
    
    # state bounds and offset
    state_high = np.array([1.5, 1.5, 5.0, 5.0])
    state_low = np.array([-1.5, -1.5, -5.0, -5.0])
    state_bounds_np = (state_high - state_low) / 2
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset = (state_high + state_low) / 2
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    state_clip_low = state_low # np.array([-1.2, -0.07])
    state_clip_high = state_high # np.array([0.6, 0.07])
    
    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.1] * env.action_space.shape[0])
    exploration_state_noise = np.array([0.01] * env.observation_space.shape[0])
    
    goal_state = np.array([0.48, 0.04, 0.0, 0.0])        # final goal state to be achived
    threshold = np.array([0.01, 0.02, 0.1, 0.1])         # threshold value to check if goal state is achieved

    # HAC parameters:
    k_level = 2                 # num of levels in hierarchy
    H = 14                      # time horizon to achieve subgoal
    lamda = 1                 # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.95                # discount factor for future rewards
    n_iter = 100                # update policy n_iter times in one DDPG update
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    
    # save trained models
    directory = "./preTrained/{}/{}level/".format(env_name, k_level) 
    filename = "HAC_{}".format(env_name)
    #########################################################
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # creating HAC agent and setting parameters
    agent = HAC(k_level, H, state_dim, action_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, lr)
    
    agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)
    
    # load agent
    agent.load(directory, filename)

    # Evaluation
    for i_episode in range(1, max_episodes+1):
        
        agent.reward = 0
        agent.timestep = 0

        state = env.reset()
        goal_state = env.unwrapped.get_obs()['desired_goal']
        goal_state = np.append(goal_state, [0.0, 0.0])
        last_state, done = agent.run_HAC(env, k_level-1, state, goal_state, True)
        if agent.check_goal(last_state, goal_state, threshold):
            print("################ Solved! ################ ")
        
        print("Episode: {}\t Reward: {}\t len: {}".format(i_episode, agent.reward, agent.timestep))
    
    env.close()


if __name__ == '__main__':
    test()
 
  

