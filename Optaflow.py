import numpy as np
import MIS
from scipy.sparse.csgraph import maximum_flow
from scipy.optimize import linear_sum_assignment as minCostMaxFlow
from scipy.sparse import csr_matrix


# given P and R, generate ANY baseline we want to outperform with r_b = minV^pi_b(s) > 0
# for instance a deterministic one should be enough (2D policies might also work, but we would need to change all 
# the policies representations to 2D)
# Probably, just look at the environment and harcode a good deterministic one with r_b > 0
def generate_baseline(P, R): 
    pass

# given P and R, generate ANY epsilon greedy policy (policy will be a 1D array representing for each state the eps-greedy action)
def generate_randEpsGreedy(P, R):
    Ns, Na = R.shape
    return np.random.choice(Na, Ns) # return an array with random actions at each state


# pol_evaluation working on both, deterministic and eps0-greedy in the finite horizon setting, not only deterministic ones
def policy_evaluation(P, R, H, policy, gamma = 0.9, tol=1e-2, eps_greedy = False, epsilon = 0.):
    """
    Args:
        P: np.array
            transition matrix (NsxNaxNs)
        R: np.array
            reward matrix (NsxNa)
        policy: np.array
            matrix mapping states to action (Ns)
        gamma: float
            discount factor
        tol: float
            precision of the solution
    Return:
        value_function: np.array
         The value function of the given policy
    """
    Ns, Na = R.shape
    value_function = np.zeros((H, Ns))

    # Deterministic case
    if not eps_greedy:
        # last step in the episode of length H, it's H-1 because of 0-index
        for s in range(Ns):
            act = policy[s]
            vs = R[s,act]
            value_function[H-1,s] = vs
        # backward induction
        for h in range(H-2, -1, -1):
            for s in range(Ns):
                act = policy[s]
                vs  = R[s,act] + np.sum(P[s,act,:]*value_function[h+1,:])
                value_function[h,s] = vs
       
    # Eps-greedy case
    else:
        # last step in the episode of length H
        for s in range(Ns):
            act = policy[s]
            vs  = 0.
            for a in range(Na):
                if a == act:
                    policy_val = 1. - epsilon*(Na-1)
                else:
                    policy_val = epsilon

                vs += policy_val*R[s,a]
            
            value_function[H-1,s] = vs
        # backward induction
        for h in range(H-2,-1,-1):
            for s in range(Ns):
                act = policy[s]
                vs  = 0.
                for a in range(Na):
                    if a == act:
                        policy_val = 1. - epsilon*(Na-1)
                    else:
                        policy_val = epsilon

                    vs += policy_val*(R[s,a] + np.sum(P[s,a,:]*value_function[h+1,:]))
                value_function[h,s] = vs
    
    return value_function
        

# compute the optimal policy and V-function but only among epsilon-0 policies
# Steps are the same, the only modification is in the policy improvement step
# I can code this one
def policy_iteration_eps0(P, R, H, gamma=0.9, tol=1e-3, epsilon = 0.):
    Ns, Na = R.shape
    policy = np.zeros(Ns, dtype = np.int) # this is an epsilon0-greedy policy
    while True:
        last_policy = policy # pi_k
        # policy evaluation
        V = policy_evaluation(P, R, H, policy, gamma, tol, eps_greedy = True, epsilon)
        # policy improvement, greedy in an epsilon0-greedy way
        qpi_values = np.argmax([R[s,:] + np.matmul(P[s,:,:], V) for s in range(Ns)])
        # TODO
        
    

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
        self.pi_0    = pi_0   # eps0-greedy policy
        self.Vpi_0   = Vpi_0  
        self.pi_b    = pi_b   # deterministic policy
        self.Vpi_b   = Vpi_b  
        self.Veps    = Vpi_opt_eps0       # only for regret computation purposes, we don't have acces to this
        self.Rb      = np.min(self.Vpi_b) # min value of V^pi_b, our assumption is that this value is positive

        self.MIS         = np.ndarray((self.T, self.S), dtype = object) # I think this should work to save
                                                                        # the S-dim array representing the policy
        self.traject_MIS = np.ndarray((self.T, self.S), dtype = object) # Same thing as above, but 3H-tuple (s_i,a_i,r_i) 
        self.rewards_MIS = np.ndarray((self.T, self.S), dtype = np.float)  # reward of the traject_MIS[t,s]

        self.Titer  = np.zeros(self.S)
        self.regret = np.zeros(self.T)

        self.saved_reward       = np.zeros(self.T)
        self.safe_saved_reward  = np.zeros((self.T, self.S))
        self.accum_saved_reward = 0.
       
        # define auxiliary attributes
        self.W      = 1./self.eps0 - self.A + 1
        self.piflow = np.ndarray((self.T, self.S), dtype = object) # policy from max flow with graph of the trajectory z_t
                                                                   # in the s-th MIS instance

    
    def V_b(self, state):
        return self.Vpi_b[state]
    
    def V_0(self, state):
        return self.Vpi_0[state]
    
    def V_eps(self, state):
        return self.Veps[state]
    
        
    def block1(self):
        self.accum_saved_reward = 0
        self.B = (1 - self.alpha)/ self.alpha * (self.S/self.Rb) * np.max(self.Vpi_b)
        
        for t in range(self.B):
            s = self.env.get_state()
            vb = self.V_b(s)
            self.accum_saved_reward += self.alpha*vb # error in the pseudocode, it is self.alpha*vb instead of vb
            if t == 0:
                # self.regret[t].append(self.V_eps(s) - vb)
                self.regret[t]   = self.V_eps(s) - vb
            else:
                # self.regret[t].append(self.regret[t-1] + self.V_eps(s) - vb)
                self.regret[t]   = self.regret[t-1] + self.V_eps(s) - vb
           
            self.saved_reward[t] = (t+1) * self.alpha * vb
            
    def block2(self):
        
        for t in range(self.B, self.T):
            s  = self.env.getstate()
            ts = self.Titer[s]
            played_baseline = True

            if ts == 0:
                policy_played   = self.pi_0
                played_baseline = False
                self.safe_saved_reward[ts,s] = (self.accum_saved_reward/self.S) + self.V_0(s) - (1 - self.alpha)*self.V_b(s)
            else:
                pi_optimist    = self.optim_step(s,ts) # pi_optimist, always eps0-greedy
                vs_theta       = self.V_robust_value(s,ts,pi_optimist)
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

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # sampling trajectory in the environment of lenght self.H starting in s, SAVE this as a np.array of H tuples (s_i,a_i,r_i)
            J # 
            a = np.random() 
            self.state = self.env.step(a) #sampled policy of length H starting from s given by policy_played, Exactly!
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if not played_baseline:
                 self.MIS[ts,s] = policy_played 
                 reward_accum_episode   = 0.
                 for h in range(self.H):
                     reward_accum_episode += J[h][2] # adding the rewards
                 
                 self.traject_MIS[ts,s] = J
                 self.rewards_MIS[ts,s] = reward_accum_episode
                 self.Titer[s]          = ts + 1
                    
            Vfunc_played   = policy_evaluation(env.P, env.R, self.H, policy_played, gamma = env.gamma, tol = 1e-6, eps_greedy = True, epsilon = self.eps0)
            # self.regret[t] = self.regret[t-1] + self.V_eps(s) - self.V(s, policy_played)
            self.regret[t] = self.regret[t-1] + self.V_eps(s) - Vfunc_played[s] # I think this is equivalent to the one above  
        
    def V_robust_value(self, s, ts, pi_0):
        Mts = self.getUppBound(s, ts)
        mu = 0
        for k in range(ts):
            denom = 0
            z_k = self.traject_MIS[k,s]
            for j in range(ts):
                denom += self.evaluatePolicy(self.MIS[j,s], z_k)
            mu += np.min(Mts, self.evaluatePolicy(pi_0, z_k)/denom) * self.rewards_MIS[k,s]
            
        return mu
    
    def getUppBound(self, s, ts):
        # return np.pow(self.W,H) * np.pow((ts / (2*np.log(ts) + 2*np.log(np.pi) + s*np.log(self.A) + np.log(1/(3*self.delta)))),(1/(1 + self.epsilon)))
        return np.inf
    
    def evaluatePolicy(self, pi_0, z_k):
        matches = 0
        for h in range(self.H):
            s, a, r = z_k[h]  
            if a == pi_0[s]: # action a is eps0-greedy wrt to s
                matches += 1
       
        # indeed, exp(log()) trick
        log_answer = matches*np.log(self.W) + self.H*np.log(self.eps0) # answer \in (0,1) -> log(answer) < 0
       
        # verify the computations are stable
        assert log_answer < 0.1 # simple verification
        if log_answer < -120:   # very small number -> probably precision errors
            print("Precision issues in EvaluatePolicy %f" % log_answer)
        
        return np.exp(log_answer) # recover answer

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # AFTER THINKING MORE, it seems flows are not necessary and "the max matching" consists only in assigning for each state s
    # the action a with the bigger weight idx(s,a) in the graph and that would be it

    def optimStep(self, s, ts): # still to check, but we need to add capacities for instance
        graph_k = self.getGraph(self.traject_MIS[ts-1,s]) # z_k: trajectory at time ts-1 in s (last trajectory inserted in s-th MIS)
        flow    = minCostMaxFlow(graph_k) # get the policy with maximal weighted match with graph z_k, the other optimal ones are already stored in piflow   
        self.piflow[ts-1,s] = flow
        return np.argmax([self.V_robust_value(s, tt, self.piflow[tt,s])] for tt in range(ts))
    
    def getGraph(self,z_k):
        idx = {}
        for h in range(self.H):
            s, a, r = z_k[h]
            try:
                idx[(s,a)] +=1
            except:
                idx[(s,a)] = 1
        
        idx  = {k: v for k, v in sorted(idx.items())}
        keys = np.array(list(idx.keys()))

        row    = keys[:,0]
        column = keys[:,1] + self.S # offset of S vertices to the actions-vertices to avoid collisions between ids
        data   = np.array(list(idx.values()))
        n      = self.S + self.A    # number of vertices in the graph, can be reduced, but OK in worst case 

        csr_graph = csr_matrix((data, (row, column)), shape=(n,n))
        return csr_graph.toarray()
            
            

