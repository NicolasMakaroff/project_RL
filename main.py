import numpy as np
import Gridworld
import Optaflow
import matplotlib.pyplot as plt
import time

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

    env = Gridworld.GridWorldWithPits(grid=grid1, txt_map=grid1_MAP, proba_succ= 1.0, 
                      uniform_trans_proba=0,normalize_reward=True) # First, let's state in a deterministic env
    H = 10 # horizon of the episode

    print(env.R.shape)
    print(env.P.shape)
    env.render()
    epsilon   = 1     # epsilon of MIS - Renyi divergence
    epsilon_0 = 1e-6  # epsilon_0 from epsilon-greedy
    
    time1 = time.time()
    pi_0  = Optaflow.generate_randEpsGreedy(env.P, env.R, H, epsilon = epsilon_0)  # Fixed eps0-greedy policy
    Vpi_0 = Optaflow.policy_evaluation(env.P, env.R, H, pi_0)                  # V^pi_0  
    
    delta = 1e-6  # just changed the name from theta to delta (same notation as pseudocode)
    T     = 600   # number of episodes, should be increased to 1e6
    pi_b  = Optaflow.generate_baseline(env.P, env.R, H) # this is the baseline, the policy we want to outperform (usually, it should be input)
                                                     # we can test our framework with a naive policy (random for instance)
    Vpi_b = Optaflow.policy_evaluation(env.P, env.R, H, pi_b) # V^pi_b
    # verify that it satisfies the assumption of r_b > 0, otherwise generate another one (maybe a while to redo it)
    print("rb according to main: ", np.min(Vpi_b[0]))
    assert np.min(Vpi_b[0]) > 0

    opt_eps0, Vpi_opt_eps0 = Optaflow.policy_iteration_eps0(env.P, env.R, H, epsilon = epsilon_0) # optimal V^pi_epsilon0
    print("Optimal e0-greedy policy:")
    # only the greedy action
    for h in range(H):
        print("Step ", h)
        for s in range(12):
            print("State ", s)
            print(np.argmax(opt_eps0[h,s]))

    alpha = 0.03   # conservative level: 0.5 is OK to test, if it works well, change to 0.05

    optaflow = Optaflow.Optaflow(env, delta, T, H, epsilon, epsilon_0, alpha, pi_0, Vpi_0, pi_b, Vpi_b, Vpi_opt_eps0) 
    optaflow.block1()
    time2 = time.time()
    optaflow.block2()
    time3 = time.time()
    print("Number of epochs T: ", T)
    print("time in Block1: ", time2 - time1)
    print("Time in Block2: ", time3 - time2)
    regret = optaflow.regret                    # regret of the first T episodes
    saved_reward = optaflow.saved_reward        # saved_reward of the first T episodes (always >= 0)
    nonCnsvProgress = optaflow.nonCnsvProgress  # Progression of non conservative episodes 
    print("Last 50 values of saved_reward: ", saved_reward[-50:])
    # Plot both, T-dim arrays: regret and saved_reward
    plt.figure()
    plt.plot(range(1,T+1), regret)
    plt.xlabel('Episode')
    plt.ylabel('Regret')
    plt.show()

    plt.figure()
    plt.plot(range(1,T+1), saved_reward)
    plt.xlabel('Episode')
    plt.ylabel('Saved Reward')
    plt.show()

    plt.figure()
    plt.plot(range(1, T+1), nonCnsvProgress)
    plt.xlabel('Episode')
    plt.ylabel('Non conservative Progress')
    plt.show()

    plt.figure()
    plt.plot(range(1, T+1), 1 + np.arange(T) - nonCnsvProgress)
    plt.xlabel('Episode')
    plt.ylabel('Conservative Progress')
    plt.show()
