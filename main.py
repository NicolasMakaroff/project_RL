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

    print(env.R.shape)
    print(env.P.shape)
    env.render()

    # ====================================================
        # YOUR IMPLEMENTATION HERE

    epsilon = 1
    epsilon_0 = 1e-6
    pi_0, V = Optaflow.policy_iteration(env.P, env.R, gamma=env.gamma, tol=1e-5)
    theta = 1e-6
    T = 1e6
    pi_b = init_policy #?
    alpha = 0.5
    optaflow = Optaflow.Optaflow(env.P, env.R, theta, T, epsilon, epsilon_0, alpha)
    optaflow.block1()
    optaflow.block2()
    regret = optaflow.regret
    # ====================================================



