import numpy as np
import MIS
from scipy.sparse.csgraph import maximum_flow


# given P and R, generate ANY baseline we want to outperform with r_b = minV^pi_b(s) > 0
# for instance a deterministic one should be enough (2D policies might also work, but we would need to change all 
# the policies representations to 2D)
# Probably, just look at the environment and harcode a good deterministic one with r_b > 0
def generate_baseline(P, R): 
    pass

# given P and R, generate ANY epsilon greedy policy (policy will be a 1D array representing for each state the eps-greedy action)
def generate_randEpsGreedy(P, R):
    pass # return an array with random action at each state, that's it


# modify pol_evaluation to work for on both, deterministic and eps0-greedy, not only deterministic ones 
# eps_greedy = False -> current code is OK
# eps_greedy = True  -> change the matrices slightly (TO DO)
def policy_evaluation(P, R, policy, gamma = 0.9, tol=1e-2, eps_greedy = False, epsilon = 0.):
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
        #=====================================================
        newP = np.zeros((Ns,Ns))
        newR = np.zeros(Ns)
        for state in range(Ns):
            add_policy = int(policy[state])
            newR[state] = R[state][add_policy]
            for nextstate in range(Ns):
                newP[state][nextstate] = P[state][add_policy][nextstate]
        value_function = np.linalg.solve(np.eye(Ns) - gamma * newP,newR)

        # ====================================================
        return value_function
    
# compute the optimal policy and V-function but only among epsilon-0 policies 
# Steps are the same, the only modification is in the policy improvement step 
# I can code this one
def policy_iteration_eps0(P, R, gamma=0.9, tol=1e-3):
    pass

# we don't use this one, I think (look more in detail)
def policy_iteration(P, R, gamma=0.9, tol=1e-3):
        """
        Args:
            P: np.array
                transition matrix (NsxNaxNs)
            R: np.array
                reward matrix (NsxNa)
            gamma: float
                discount factor
            tol: float
                precision of the solution
        Return:
            policy: np.array
                the final policy
            V: np.array
                the value function associated to the final policy
        """
    Ns, Na = R.shape
    V = np.zeros(Ns)
    policy = np.zeros(Ns, dtype=np.int)
    # ====================================================
    
    V_old = V
    V = policy_evaluation(P,R,policy,gamma,tol)

    converged = False
    while (not converged):

        V = R + gamma * P.dot(V)
        greedy_policy = np.argmax(V, axis = 1)
        if np.all(greedy_policy == policy):
            converged = True
        policy = greedy_policy
        V = policy_evaluation(P,R,policy,gamma,tol)
    # ====================================================
    return policy, V

class Optaflow():
    
    def __init__(self, env, delta, T, epsilon, epsilon_0, alpha, pi_0, Vpi_0, pi_b, Vpi_b, Vpi_opt_eps0):
        self.B   = None
        self.env = env
        self.H   = env.H    # horizon of the environment, H should be included in the environment
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
        self.traject_MIS = np.ndarray((self.T, self.S), dtype = object) # Same thing as above, but H-tuple (or 2H)

        self.Titer  = np.zeros(self.S)
        self.regret = np.zeros(self.T)

        self.saved_reward       = np.zeros(self.T)
        self.safe_saved_reward  = np.zeros((self.T, self.S))
        self.accum_saved_reward = 0.
       
        # define auxiliary attributes
        self.W = 1./self.eps0 - self.A + 1

    
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
            # sampling trajectory in the environment of lenght self.H (env.H) starting in s
            J # 
            a = np.random() 
            self.state = self.env.step(a) #sampled policy of length H starting from s given by policy_played, Exactly!
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if not played_baseline:
                 self.MIS[ts,s] = policy_played 
                 self.traject_MIS[ts,s] = J
                 self.Titer[s] = ts + 1
                    
            Vfunc_played   = policy_evaluation(env.P, env.R, policy_played, gamma = env.gamma, tol = 1e-6, eps_greedy = True, epsilon = self.eps0)
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
            mu += np.min(Mts, self.evaluatePolicy(pi_0, z_k)/denom) * self.reward(z_k) # typo in the notes Mts, DON'T forget to code self.reward
            
        return mu
    
    def getUppBound(self, s, ts):
        # return np.pow(self.W,H) * np.pow((ts / (2*np.log(ts) + 2*np.log(np.pi) + s*np.log(self.A) + np.log(1/(3*self.delta)))),(1/(1 + self.epsilon)))
        return np.inf
    
    def evaluatePolicy(self, pi_0, z_k):
        matches = 0
        for i in range(self.H):
            s, a = z_k  # z_k[2*i, 2*i+1] maybe, depends on how you save the trajectory
            if a == pi_0[s]: # action a is eps0-greedy wrt to s
                matches += 1
       
        # indeed, exp(log()) trick
        log_answer = matches*np.log(self.W) + self.H*np.log(self.eps0) # answer \in (0,1) -> log(answer) < 0
       
        # verify the computations are stable
        assert log_answer < 0.1 # simple verification
        if log_answer < -120:   # very small number -> probably precision erros
            print("Precision issues in EvaluatePolicy %f" % log_answer)
        
        return np.exp(log_answer) # recover answer

    def optimStep(self, s, ts): # still to check, but we need to add capacities for instance
        for k in range(ts):
            graph_k, init_vertex, sink_vertex = self.getGraph(z_k)
            value_flow, res_graph = maximum_flow(graph_k, init_vertex, sink_vertex) #O(V.E^2)
            
        k_0 = np.argmax(self.V_robust_value(s, ts, res_graph))
            
        return k_0

    
    
            
            

