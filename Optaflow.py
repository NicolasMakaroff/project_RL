import numpy as np
import MIS
from scipy.sparse.csgraph import maximum_flow

def policy_evaluation(self, P, R, policy, gamma = 0.9, tol=1e-2):
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
    
    def __init__(self, env, theta, T, epsilon, epsilon_0, alpha):
        self.regret = []
        self.saved_reward = []
        self.B = None
        self.env = env
        self.accum_saved_reward = []
        self.theta = theta
        self.A = None
        self.epsilon = epsilon
        self.eps0 = epsilon_0
        self.T = T
        self.alpha = alpha
        self.MIS = []
        self.traject_MIS = []
        
    
    def V_b(self):
        pass
    
    def V_0(self):
        pass
    
    def V_eps(self):
        pass
    
        
    def block1(self):
        self.accum_saved_reward = 0
        self.B = (1 - self.alpha)/ self.alpha * (S/Rb) * self.Vb(s)
        
        for t in range(B):
            s = self.env.get_state()
            vb = self.V_b(s)
            self.accum_saved_reward += vb
            if t == 0:
                self.regret[t].append(self.V_eps(s) - vb)
            else:
                self.regret[t].append(self.regret[t-1] + self.V_eps(s) - vb)
            self.saved_reward[t] = (t+1) * self.alpha * vb
            
    def block2(self):
        
        for t in range(B,T): #????
            s = self.env.getstate()
            ts = Titer[s]
            played_baseline = True
            if ts == 0:
                policy_played = pi_0
                played_baseline = False
                safe_saved_reward[ts,s] = (self.accum_saved_reward/S) + self.V_0(s) - (1 - self.alpha)*self.V_b(s)
            else:
                pi_0 = self.optim_step(s,ts)
                vs_theta = self.V_robust_value(s,ts,pi_0)
                a1 = safe_saved_reward[ts-1,s] + vs_theta
                a2 = (1 - alpha) * self.V_b(s)
                if a1 >= a2:
                    policy_played = pi_0
                    played_baseline = False
                    safe_saved_reward[ts,s] = a1 - a2
                else:
                    policy_played = self.V_b(s)
                    played_baseline = True
                    safe_saved_reward[ts,s] = safe_saved_reward[ts-1,s] + self.alpha * self.V_b(s)
                    
            for s in self.env.state:
                saved_reward[t] = safe_saved_reward[Titer[s],s]
                J # =
                a = np.random() 
                self.state = self.env.step(a) #sampled policy of length H starting from s given by policy_played
                if not played_baseline:
                    self.MIS[ts,s] = policy_played
                    self.traject_MIS[ts,s] = J
                    Titer[s] = ts + 1
                    
                self.regret[t] = self.regret[t-1] + self.V_eps(s) - self.V(s, policy_played)
                
        
    def V_robust_value(self, s, ts, pi_0):
        Mts = self.getUppBound(s, ts)
        mu = 0
        for k in range(ts):
            denom = 0
            z_k = self.traject_MIS[k,s]
            for j in range(ts):
                denom += self.evaluatePolicy(self.MIS[j,s], z_k)
            mu += np.min(Ts, self.evaluatePolicy(pi_0, z_k)/denom) * self.reward(z_k)
            
        return mu
    
    def getUppBound(self, s, ts):
        #return np.pow(self.W,H) * np.pow((ts / (2*np.log(ts) + 2*np.log(np.pi) + s*np.log(self.A) + np.log(1/(3*self.theta)))),(1/(1 + self.epsilon)))
        return np.inf
    
    def evaluatePolicy(self, pi_0, z_k):
        matches = 0
        for i in range(H):
            s, a = z_k
            if a == self.V_0(s):
                matches += 1
        return np.pow(W,matches) * np.pow(self.eps0,H) #passer au log ?
        
    def optimStep(self, s, ts):
        for k in range(ts):
            graph_k, init_vertex, sink_vertex = self.getGraph(z_k)
            value_flow, res_graph = maximum_flow(graph_k, init_vertex, sink_vertex) #O(V.E^2)
            
        k_0 = np.argmax(self.V_robust_value(s, ts, res_graph))
            
        return k_0
    
    def getGraph(self,z_k):
        idx = {}
        for i in z_k:
            (s,a) = i
            try:
                idx[(s,a)] +=1
            except:
                idx[(s,a)] = 1
        idx = {k: v for k, v in sorted(idx.items())}
        keys = np.array(list(idx.keys()))
        no_double = list(dict.fromkeys(keys[:,0]))
        row = keys[:,0]#np.array([0, 1, 3, 4])
        column = keys[:,1]
        data = np.array(list(idx.values()))
        n = max(column.max()+1, row.max()+1)
        csr_graph = csr_matrix((data, (row, column)), shape=(n,n))
        return csr_graph.toarray()
            
            