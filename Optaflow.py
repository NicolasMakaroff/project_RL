import numpy as np
import math
import MIS
from scipy.sparse.csgraph import maximum_flow
from scipy.optimize import linear_sum_assignment as minCostMaxFlow
from scipy.sparse import csr_matrix


# given P, R and H, generate ANY baseline we want to outperform with r_b = minV^pi_b(s) > 0
# We hardcode the policy given in the paper
# The policy is composed of H decision rules, each of them in this case is the same
def generate_baseline(P, R, H):
    """
    Args: 
        P: transition matrix
        R: reward matrix
        H: horizon

    Return:
        policy_baseline: policy given in the AISTATS paper replicated H times (for each decision rule)
    """
    # Harcoding the baseline given by Evrard in the AISTATS paper
    ### MAP
    # 0  1  2  3
    # 4  5  6  7
    # 8  9 10 11

    # Actions, > (0), v (1), < (2), ^ (3)
    Ns, Na  = R.shape
    pol     = np.zeros((Ns,Na))

    pol[0]  = np.array([0.,1.,0.,0.]) # down
    pol[1]  = np.array([0.,0.,1.,0.]) # left
    pol[2]  = np.array([.5,0.,.5,0.]) # left or right
    pol[3]  = np.array([1.,0.,0.,0.]) # we are in the goal G, here it doesn't matter

    pol[4]  = np.array([0.,1.,0.,0.]) # down
    pol[5]  = np.array([1.,0.,0.,0.]) # This is the trap X, seems not be a terminal state
    pol[6]  = np.array([0.,0.,0.,1.]) # up
    pol[7]  = np.array([0.,0.,.5,.5]) # left or up

    pol[8]  = np.array([1.,0.,0.,0.]) # right, initial state S
    pol[9]  = np.array([1.,0.,0.,0.]) # right
    pol[10] = np.array([.5,0.,0.,.5]) # up or right
    pol[11] = np.array([0.,0.,0.,1.]) # up

    # The decision rule at each step h is the same (pol described above)
    policy_b = np.zeros((H,Ns,Na))
    for h in range(H):
        policy_b[h] = pol
    
    return policy_b

# given P, R, H and epsilon, generate ANY epsilon greedy policy 
def generate_randEpsGreedy(P, R, H, epsilon = 0.):
    """
    Args:
        P: transition matrix
        R: reward matrix
        H: horizon
        epsilon: Pi_{epsilon-soft} space for policies

    Return:
        policy_random: policy with H decision rules, each of them is a random e-greedy policy
    """
    Ns, Na        = R.shape
    policy_random = np.zeros((H, Ns, Na))
    for h in range(H):
        # return an array with random actions at each state
        random_decision  = np.random.choice(Na,Ns)
        policy_random[h] = epsilon*np.ones((Ns,Na))
        for s in range(Ns):
            act = random_decision[s]
            policy_random[h,s,act] = 1-epsilon*(Na-1)

    return policy_random # policy with H decision rules, each of them is a random epsilon-greedy policy


# policy is composed of H decision rules (d_1, d_2, ..., d_H) each of them is a policy (d_i:SxA ->[0,1])
# policy[h,s,a] is the value of decision rule d_h(s,a)
def policy_evaluation(P, R, H, policy):
    """
    Args:
        P: transition matrix          (NsxNaxNs)
        R: reward matrix              (NsxNa)
        H: horizon
        policy: finite horizon policy (HxNsxNa)
        
    Return:
        value_function: np.array
         The value function of the given policy
    """
    Ns, Na = R.shape
    value_function = np.zeros((H, Ns))

    # last step in the episode length H, it's H-1 because of 0-index
    for s in range(Ns):
        value_funct = 0.
        for a in range(Na):
            value_funct += policy[H-1,s,a]*R[s,a]
        value_function[H-1,s] = value_funct

    # backward induction
    for h in range(H-2,-1,-1):
        for s in range(Ns):
            value_funct = 0.
            for a in range(Na):
                value_funct += policy[h,s,a]*(R[s,a] + np.sum(P[s,a,:]*value_function[h+1,:]))
            
            value_function[h,s] = value_funct 
    
    return value_function
        

# compute the optimal policy and V-function but only among epsilon-0 policies
def policy_iteration_eps0(P, R, H, epsilon = 0.):
    """
    Args: 
        P: transition matrix                 (NsxNaxNs)
        R: reward matrix                     (NsxNa)
        H: horizon                      
        policy: finite horizon policy        (HxNsxNa)
        epsilon: Pi_{epsilon-soft} space     0 <= epsilon <= 1/Na

    Return:
        value_function: 
         The value function of the given policy
    """
    Ns, Na = R.shape
    policy_opt = np.zeros((H, Ns, Na)) # this will be the optimal epsilon-greedy policy
    value_opt  = np.zeros((H,Ns))      # this will be the optimal value function of the policy above
    
    # last step in the epsilode length H
    for s in range(Ns):
        Q_Hs                = R[s,:]
        opt_act             = np.argmax(Q_Hs)
        # choosing the H-th decision rule for state s as an epsilon-greedy policy
        rule_sH             = epsilon*np.ones(Na)
        rule_sH[opt_act]    = 1 - epsilon*(Na-1)
        policy_opt[H-1,s,:] = rule_sH
        value_opt[H-1,s]    = np.sum(rule_sH*Q_Hs)

    # bacward induction
    for h in range(H-2,-1,-1):
        for s in range(Ns):
            Qh_s = np.zeros(Na)
            for a in range(Na):
                Qh_s[a] = R[s,a] + np.sum(P[s,a,:]*value_opt[h+1,:])
            
            opt_act = np.argmax(Qh_s)
            # choosing the h-th decision rule
            rule_sh           = epsilon*np.ones(Na)
            rule_sh[opt_act]  = 1 - epsilon*(Na-1)
            policy_opt[h,s,:] = rule_sh
            value_opt[h,s,:]  = np.sum(rule_sh*Qh_s)

    return policy_opt, value_opt


class Optaflow():
    
    def __init__(self, env, delta, T, H, epsilon, epsilon_0, alpha, pi_0, Vpi_0, pi_b, Vpi_b, Vpi_opt_eps0):
        self.B   = None
        self.env = env
        self.H   = H        # horizon H of the episoden
        self.delta = delta  # seemingly not useful anymore (due to inf in getUppBound)
        self.A     = env.Na # number of actions
        self.S     = env.Ns # number of states
        self.epsilon = epsilon
        self.eps0    = epsilon_0
        self.T       = T
        self.alpha   = alpha
        # policies and V-functions from precomputation
        self.pi_0    = pi_0   # eps0-greedy policy     (HxNsxNa)
        self.Vpi_0   = Vpi_0  # value function of pi_0 (HxNs)
        self.pi_b    = pi_b   # baseline policy        (HxNsxNa)
        self.Vpi_b   = Vpi_b  # value function of pi_b (HxNs)
        self.Veps    = Vpi_opt_eps0       # only for regret computation purposes, we don't have acces to this
        self.Rb      = np.min(self.Vpi_b) # min value of V^pi_b, our assumption is that this value is positive

        self.MIS         = np.ndarray((self.T, self.S), dtype = object)    # I think this should work to save
                                                                           # the policy in the s-th instance
        self.traject_MIS = np.ndarray((self.T, self.S), dtype = object)    # Same thing as above, but 3H-tuple (s_i,a_i,r_i) 
        self.rewards_MIS = np.ndarray((self.T, self.S), dtype = np.float)  # reward of the traject_MIS[t,s]

        self.Titer  = np.zeros(self.S) # counter of the number of samples t in the s-th MIS
        self.regret = np.zeros(self.T) # regret at instant T over all the states

        self.saved_reward       = np.zeros(self.T)
        self.safe_saved_reward  = np.zeros((self.T, self.S))
        self.accum_saved_reward = 0.
       
        # define auxiliary attributes
        self.W        = 1./self.eps0 - self.A + 1
        self.piflow   = np.ndarray((self.T, self.S), dtype = object) # policy from max flow with graph of the trajectory z_t
                                                                     # in the s-th MIS instance
        self.cumDenom = np.zeros((self.S,self.T,self.T))             # cumDenom(k,r) = p_{x_k}(z_r) + ... + p_{x_0}(z_r) in the s-th MIS

        self.env.reset()                                             # this should not be necessary, but initialize just in case

    # V-function (expected reward) for each policy and state at instant 0, before sampling the trajectory
    def V_b(self, state):
        return self.Vpi_b[0, state]
    
    def V_0(self, state):
        return self.Vpi_0[0, state]
    
    def V_eps(self, state):
        return self.Veps[0, state]
    
        
    # Initialization of the strategy. Here, we save reward to initialize the s MIS instances and deploy OPTaFlow algorithm
    def block1(self):
        self.accum_saved_reward = 0
        self.B = int(math.ceil((1 - self.alpha)/ self.alpha * (self.S/self.Rb) * np.max(self.Vpi_b)))
        
        for t in range(self.B):
            s  = self.env.curState() # observe the state, you know it in advance at the beginning of each episode (don't take actions)
            vb = self.V_b(s)
            self.accum_saved_reward += self.alpha*vb
            if t == 0:
                self.regret[t]   = self.V_eps(s) - vb
            else:
                self.regret[t]   = self.regret[t-1] + self.V_eps(s) - vb
           
            self.saved_reward[t] = (t+1) * self.alpha * vb

            # In theory, we follow the baseline policy here and should sample a H-length trajectory, too
            # It won't change anything, but let's pretend we do it
            self.env.reset() # only the reset of the end of the episode


    # Actual algorithm, starts after B episodes (nb of episodes where we only saved reward)        
    def block2(self):
        
        for t in range(self.B, self.T):
            s  = self.env.curState() # observe the episode, you know it in advance
            ts = self.Titer[s]
            played_baseline = True

            if ts == 0:
                policy_played   = self.pi_0
                played_baseline = False
                self.safe_saved_reward[ts,s] = (self.accum_saved_reward/self.S) + self.V_0(s) - (1 - self.alpha)*self.V_b(s)
            else:
                pi_optimist    = self.optimStep(s,ts)                     # pi_optimist, always eps0-greedy
                vs_theta       = self.V_robust_value(s,ts,pi_optimist)     # lower_bound given by OPTIMIST for V^pi_optimist[0,s]
                expPerformance = self.safe_saved_reward[ts-1,s] + vs_theta # what you think will win with this policy
                minPerformance = (1 - self.alpha) * self.V_b(s)            # what you need to outperform
                if expPerformance >= minPerformance:
                    policy_played   = pi_optimist
                    played_baseline = False
                    self.safe_saved_reward[ts,s] = self.safe_saved_reward[ts-1,s] + expPerformance - minPerformance
                else:
                    policy_played   = self.pi_b
                    played_baseline = True
                    self.safe_saved_reward[ts,s] = self.safe_saved_reward[ts-1,s] + self.alpha * self.V_b(s)
                    
            for s0 in self.env.state:
                self.saved_reward[t] += self.safe_saved_reward[self.Titer[s0],s0]


            # sampling trajectory in the environment of lenght self.H starting in s -> list of lenght H in the format (s_i,a_i,r_i)
            J = []
            for h in range(self.H):
                st = env.curState()
                # sample using the h-th decision rule policy[h]
                act = np.random.choice(self.A, 1, p = policy_played[h,st]).item()
                next_st, reward, done, _ = self.env.step(act)
                J.append((st, act, reward))
                if done:
                    self.env.reset()

            self.env.reset() # end of the episode

            # update MIS instance
            if not played_baseline:
                 self.MIS[ts,s] = policy_played
                 reward_accum_episode   = 0.
                 for h in range(self.H):
                     reward_accum_episode += J[h][2] # adding the rewards
                 
                 self.traject_MIS[ts,s] = J
                 self.rewards_MIS[ts,s] = reward_accum_episode

                 # update cumDenom: cumDenom(k,r) = cumDenom(k-1,r) + p_{x_k}(z_r)
                 # cumDenom(i, ts) and cumDenom(ts, i) where 0 <= i <= ts, all the other values are already filled
                 self.cumDenom[s,0,ts] = self.evaluatePolicy(self.MIS[0,s], J)
                 self.cumDenom[s,ts,0] = (ts>0)*self.cumDenom[s,ts-1,0] + self.evaluatePolicy(policy_played, self.traject_MIS[0,s])
                 for k0 in range(1,ts+1):
                     self.cumDenom[s,k0,ts] = self.cumDenom[s,k0-1,ts] + self.evaluatePolicy(self.MIS[k0,s], J)
                     self.cumDenom[s,ts,k0] = self.cumDenom[s,ts-1,k0] + self.evaluatePolicy(policy_played, self.traject_MIS[k0,s])

                 self.Titer[s] = ts + 1
                    
            Vfunc_played   = policy_evaluation(env.P, env.R, self.H, policy_played)
            self.regret[t] = self.regret[t-1] + self.V_eps(s) - Vfunc_played[0,s]  
        
    def V_robust_value(self, s, ts, pi):
        """
            Args:
                s: s-th MIS instance
                ts: iteration t in the s-th MIS instance
                pi: policy to evaluate (HxNsxNa)
            
            Return:
                mu_t: MIS estimator of V^pi[0,s] computed using the expression of MIS paper
        """
        Mts = self.getUppBound(s, ts)
        mu = 0
        for k in range(ts):
            denom = 0
            z_k = self.traject_MIS[k,s]
            for j in range(ts):
                denom += self.evaluatePolicy(self.MIS[j,s], z_k)
            mu += np.min(Mts, self.evaluatePolicy(pi, z_k)/denom) * self.rewards_MIS[k,s]
            
        return mu
    
    def getUppBound(self, s, ts):
        """
            Args: 
                s: s-th MIS instance
                ts: iteration t in the s-th MIS instance

            Return:
                Mt: truncatated value for the robust estimator as in the MIS paper
        """
        # return np.pow(self.W,H) * np.pow((ts / (2*np.log(ts) + 2*np.log(np.pi) + s*np.log(self.A) + np.log(1/(3*self.delta)))),(1/(1 + self.epsilon)))
        return np.inf
    
    def evaluatePolicy(self, pi, z_k):
        """
            Args:
                pi: policy (HxNsxNa)
                z_k: trajectory of length H (an episode)

            Return:
                answer: pi(z_k) considering only the terms due to the policy pi and not to the dynamic p of the MDP
        """
        matches = 0
        for h in range(self.H):
            s, a, r = z_k[h]  
            if a == np.argmax(pi[h,s]): # action a is eps0-greedy wrt to s in the h-th decision rule pi_0[h]
                matches += 1
       
        # indeed, exp(log()) trick
        log_answer = matches*np.log(self.W) + self.H*np.log(self.eps0) # answer \in (0,1) -> log(answer) < 0
       
        # verify the computations are stable
        assert log_answer < 0.1 # simple verification
        if log_answer < -120:   # very small number -> probably precision errors
            print("Precision issues in EvaluatePolicy %f" % log_answer)
        
        return np.exp(log_answer) # recover answer

    def optimStep(self, s, ts):
        """
            Args:
                s: s-th MIS instance
                ts: iteration t in the s-th MIS instance
            Return:
                greedy_policy: suggested by greedy OPTIMIST

            Logic:
            The strategy is the following:
            u_nonRobust = \sum_k p_x(z_k) * R(z_k) / cumDenom(ts-1,k)
        
            At each step, sort R(z_k)/cumDenom(ts-1,k) and take the max, make the policy x match the trajectory z_k (in the graph interpretation way)
            Notice the policy x consists of H decision rules and therefore, at most one state is updated at each step h \in [1,H]

            With the new policy x, remove the previous matched trajectory and repeat the process (only update the states of x non previously updated
            with the new trajectories)
        """
        greedy_policy = -1*np.ones((self.H,self.S,self.A)) # H decision rules
        active_traj   = {key: 1 for key in range(ts)}
        while True: # at most ts iterations (each iteration removes 1 element from active_traj)
            # policy is full or no more trajectories are available
            policy_full  = np.min(greedy_policy) >= 0
            more_traject = len(active_traj) > 0
            
            if (not more traject) or policy_full:
                break

            potential_val_max = -1
            idx_maxPot        = -1
            for k0 in active_traj:
                rk = self.rewards_MIS[k0,s]
                dk = self.cumDenom[s,ts-1,k0]
                zk = self.traject_MIS[k0,s]
                max_matches = 0
                for h in range(self.H):
                    s, a, r = zk[h]
                    chosen_act  = np.min(greedy_policy[h,s]) >= 0 
                    cur_opt_act = np.argmin(greedy_policy[h,s]) 
                    if (not chosen_act) or cur_opt_act == a: # chosen action equal to proposed action "a"  or not chosen gives you a possible match 
                        max_matches += 1

                max_potential = np.exp(max_matches.np.log(self.W) + self.H*np.log(self.eps0))

                if max_potential >= potential_val_max:
                    potential_val_max = max_potential
                    idx_maxPot        = k0

            # be greedy wrt the max potential val (max matching) in all the decision rules
            zk_opt = self.traject_MIS[idx_maxPot,s]
            for h in range(self.H):
                s, a, r = zk_opt[h]
                chosen_act = np.min(greedy_policy[h,s]) >= 0
                if not chosen_act:
                    # set the action given by zk_opt as the optimal in the h-th decision rule
                    greedy_policy[h,s]   = self.eps0*np.ones(self.A)
                    greedy_policy[h,s,a] = 1-self.eps0*(self.A-1)
            
            # update active_traj and sort again with the remaining trajectories and the updated policy
            del active_traj[idx_maxPot]

        # policy is fully updated (usually, not the case) or all trajectories traversed (more likely) -> complete randomly the policy
        for h in range(self.H):
            for s in range(self.S):
                chosen_act = np.min(greedy_policy[h,s]) >= 0
                if not chosen_act:
                    greedy_policy[h,s]   = self.eps0*np.ones(self.A)
                    greedy_policy[h,s,0] = 1-self.eps0*(self.A-1)   # always completing with action 0 (right ->) 

        return greedy_policy

