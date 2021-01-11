import numpy as np
import Gridworld
import Optaflow

if __name__ == '__main__':
    
    grid1 = [
            ['', '', '','g',],
            ['', 'x', '', '',],
            ['s', '', '', '',]
        ]
    grid1_MAP = [
            "+-------+",
            "| : : :G|",
            "| :x: : |",
            "|S: : : |",
            "+-------+",
        ]

    env = GridWorldWithPits(grid=grid1, txt_map=grid1_MAP, proba_succ= 0.95, 
                      uniform_trans_proba=0,normalize_reward=True)   
    H = 10 # horizon of the episode

    print(env.R.shape)
    print(env.P.shape)
    env.render()

    # ====================================================
    epsilon   = 1     # epsilon of MIS - Renyi divergence
    epsilon_0 = 1e-6  # epsilon_0 from epsilon-greedy
    
    pi_0   = Optaflow.generate_randEpsGreedy(env.P, env.R, eps = epsilon_0)  # Fixed eps0-greedy policy
    Vpi_0  = Optaflow.policy_evaluation(env.P, env.R, H, pi_0, gamma = env.gamma, eps_greedy = True, epsilon = epsilon_0) # V^pi_0  
    
    delta = 1e-6  # just changed the name from theta to delta (same notation as pseudocode)
    T     = 1e6   # number of episodes
    pi_b  = Optaflow.generate_baseline(env.P, env.R) # this is the baseline, the policy we want to outperform (usually, it should be input)
                                                     # we can test our framework with a naive policy (random for instance)
    Vpi_b = Optaflow.policy_evaluation(env.P,env.R, H, pi_b, gamma = env.gamma, eps_greedy = False) # V^pi_b
    # verify that it satisfies the assumption of r_b > 0, otherwise generate another one (maybe a while to redo it)
    assert np.min(Vpi_b) > 0

    opt_eps0, Vpi_opt_eps0 = Optaflow.policy_iteration_eps0(env.P, env.R, H, gamma = env.gamma, tol = 1e-5, epsilon = epsilon_0) # optimal V^pi_epsilon0
    # print("Optimal e0-greedy policy: ", opt_eps0)

    alpha = 0.5   # conservative level: 0.5 is OK to test, if it works well, change to 0.9
   
    optaflow = Optaflow.Optaflow(env.P, env.R, delta, T, H, epsilon, epsilon_0, alpha, pi_0, Vpi_0, pi_b, Vpi_b, Vpi_opt_eps0) 
    optaflow.block1()
    optaflow.block2()
    regret = optaflow.regret              # regret of the first T episodes
    saved_reward = optaflow.saved_reward  # saved_reward of the first T episodes (always >= 0) 
    # =====================================================
    # Plot both, T-dim arrays: regret and saved_reward
