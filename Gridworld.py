from abc import ABC, abstractmethod
import numpy as np
import copy
import sys
import time
from gym import utils


class FiniteEnv(ABC):
    """
    Base class for a finite MDP.

    Args:
        states      (list): List of legnth S containing the indexes of the states, e.g. [0,1,2]
        action_sets (list): List containing the actions available in each state, e.g. [[0,1], [2,3]],
                            action_sets[i][j] returns the index of the j-th available action in state i
        P       (np.array): Array of shape (Ns, Na, Ns) containing the transition probabilities,
                            P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a). Na is the total number of actions.
        gamma      (float): discount factor


    Attributes:
        Ns   (int): Number of states
        Na   (int): Number of actions
        actions (list): list containing all possible actions = [0, 1, ..., Na-1]

    """
    def __init__(self, states, action_sets, P, gamma):

        self.states = states
        self.action_sets = action_sets
        self.actions = list(set().union(*action_sets))
        self.Ns = len(states)
        self.Na = len(self.actions)
        self.P = P

        self.state = 0  # initial state
        self.gamma = gamma
        self.reset()
        super().__init__()

    def available_actions(self, state=None):
        """
        Return all actions available in a given state.
        """
        if state is not None:
            return self.action_sets[state]
        return self.action_sets[self.state]

    @abstractmethod
    def reset(self):
        """
        Reset the environment to a default state.

        Returns:
            state (object)
        """
        pass

    @abstractmethod
    def curState(self):
        """
        Observe current state of the environment (no action is taken)

        Return:
            state (object)
        """
        pass

    @abstractmethod
    def reward_func(self, state, action, next_state):
        """
        Args:
            state      (int): current state
            action     (int): current action
            next_state (int): next state

        Returns:
            reward (float)
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Execute a step. Similar to gym function [1].
        [1] https://gym.openai.com/docs/#environments

        Args:
            action (int): index of the action to take

        Returns:
            observation (object)
            reward      (float)
            done        (bool)
            info        (dict)
        """
        pass

    @abstractmethod
    def sample_transition(self, s, a):
        """
        Sample a transition s' from P(s'|s,a).

        Args:
            s (int): index of state
            a (int): index of action

        Returns:
            ss (int): index of next state
        """
        pass

class GridWorldWithPits(FiniteEnv):
    def __init__(self, grid, txt_map, gamma=0.99, proba_succ=0.95, uniform_trans_proba=0.001, normalize_reward=False):
        self.desc = np.asarray(txt_map, dtype='c')
        self.grid = grid
        self.txt_map = txt_map

        self.action_names = np.array(['right', 'down', 'left', 'up'])

        self.n_rows, self.n_cols = len(self.grid), max(map(len, self.grid))

        # Create a map to translate coordinates [r,c] to scalar index
        # (i.e., state) and vice-versa
        self.normalize_reward = normalize_reward


        self.initial_state = None
        self.coord2state = np.empty_like(self.grid, dtype=np.int)
        self.nb_states = 0
        self.state2coord = []
        for i in range(self.n_rows):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != 'w':
                    if self.grid[i][j] == 's':
                        self.initial_state = self.nb_states
                    self.coord2state[i, j] = self.nb_states
                    self.nb_states += 1
                    self.state2coord.append([i, j])
                else:
                    self.coord2state[i, j] = -1

        self.P = None
        self.R = None
        self.proba_succ = proba_succ
        self.uniform_trans_proba = uniform_trans_proba

        # compute the actions available in each state
        self.state_actions = [range(len(self.action_names)) for _ in range(self.nb_states)]
        # self.state_actions = self.compute_available_actions(), sometimes this might be necessary, but for us, not calling it is ok
        self.matrix_representation()
        self.lastaction = None
        super(GridWorldWithPits, self).__init__(states=range(self.nb_states), action_sets=self.state_actions, P=self.P, gamma=gamma)
        self.current_step = 0

    def matrix_representation(self):
        if self.P is None:
            nstates = self.nb_states
            nactions = max(map(len, self.state_actions))
            self.P = np.inf * np.ones((nstates, nactions, nstates))
            self.R = np.inf * np.ones((nstates, nactions))
            for s in range(nstates):
                r, c = self.state2coord[s]
                for a_idx, action in enumerate(range(len(self.action_names))):
                    self.P[s, a_idx].fill(0.)
                    if self.grid[r][c] == 'g':
                        # self.P[s, a_idx, self.initial_state] = 1.
                        self.P[s, a_idx, s] = 1. # once you reach a final state, continue there
                        self.R[s, a_idx] = 10.
                    
                    else:
                        ns_succ, ns_fail = np.inf, np.inf
                        if action == 0:
                            ns_succ = self.coord2state[r, min(self.n_cols - 1, c + 1)]
                            ns_fail = [self.coord2state[r, max(0, c - 1)],
                            self.coord2state[min(self.n_rows - 1, r + 1), c],
                            self.coord2state[max(0, r - 1), c]
                            ]

                        elif action == 1:
                            ns_succ = self.coord2state[min(self.n_rows - 1, r + 1), c]
                            ns_fail = [self.coord2state[max(0, r - 1), c],
                            self.coord2state[r, max(0, c - 1)],
                            self.coord2state[r, min(self.n_cols - 1, c + 1)]
                            ]
                        elif action == 2:
                            ns_succ = self.coord2state[r, max(0, c - 1)]
                            ns_fail = [self.coord2state[r, min(self.n_cols - 1, c + 1)],
                            self.coord2state[max(0, r - 1), c],
                            self.coord2state[min(self.n_rows - 1, r + 1), c]
                            ]
                        elif action == 3:
                            ns_succ = self.coord2state[max(0, r - 1), c]
                            ns_fail = [self.coord2state[min(self.n_rows - 1, r + 1), c],
                            self.coord2state[r, min(self.n_cols - 1, c + 1)],
                            self.coord2state[r, max(0, c - 1)]
                            ]

                        L = []
                        for el in ns_fail:
                            x, y = self.state2coord[el]
                            if self.grid[x][y] == 'w':
                                L.append(s)
                            else:
                                L.append(el)

                        self.P[s, a_idx, ns_succ] = self.proba_succ
                        for el in L:
                            self.P[s, a_idx, el] += (1. - self.proba_succ)/len(ns_fail)
                        # self.P[s, a_idx] = self.P[s, a_idx] + self.uniform_trans_proba / nstates
                        # self.P[s, a_idx] = self.P[s, a_idx] / np.sum(self.P[s, a_idx])

                        assert np.isclose(self.P[s, a_idx].sum(), 1)

                        if self.grid[r][c] == 'x':
                            self.R[s, a_idx] = -20
                        else:
                            self.R[s, a_idx] = -2

            if self.normalize_reward:
                minr = np.min(self.R)
                maxr = np.max(self.R[np.isfinite(self.R)])
                self.R = (self.R - minr) / (maxr - minr)

            self.d0 = np.zeros((nstates,))
            self.d0[self.initial_state] = 1.

    def compute_available_actions(self):
        # define available actions in each state
        # actions are indexed by: 0=right, 1=down, 2=left, 3=up
        state_actions = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.grid[i][j] == 'g':
                    state_actions.append([0])
                elif self.grid[i][j] != 'w':
                    actions = [0, 1, 2, 3]
                    if i == 0:
                        actions.remove(3)
                    if j == self.n_cols - 1:
                        actions.remove(0)
                    if i == self.n_rows - 1:
                        actions.remove(1)
                    if j == 0:
                        actions.remove(2)

                    for a in copy.copy(actions):
                        r, c = i, j
                        if a == 0:
                            c = min(self.n_cols - 1, c + 1)
                        elif a == 1:
                            r = min(self.n_rows - 1, r + 1)
                        elif a == 2:
                            c = max(0, c - 1)
                        else:
                            r = max(0, r - 1)
                        if self.grid[r][c] == 'w':
                            actions.remove(a)

                    state_actions.append(actions)
        return state_actions

    def description(self):
        desc = {
            'name': type(self).__name__
        }
        return desc

    def reward_func(self, state, action, next_state):
        return self.R[state, action]
    
    def curState(self):
        return self.state

    def reset(self, s=None):
        self.lastaction = None
        if s is None:
            self.state = self.initial_state
        else:
            self.state = s
        self.current_step = 0
        return self.state

    def step(self, action):
        try:
            action_index = self.state_actions[self.state].index(action)
        except:
            raise ValueError("Action {} cannot be executed in this state {}".format(action, self.state))

        p = self.P[self.state, action_index]
        next_state = np.random.choice(self.nb_states, 1, p=p).item()

        reward = self.R[self.state, action_index]

        self.lastaction = action

        r, c = self.state2coord[self.state]
        # I think we should return done if we fall in the trap too and restart the episode
        done = self.grid[r][c] == 'g'
        # We continue in the terminal state and don't sample the next state
        # if done:
        #    next_state = self.state # continue in the terminal state

        # done = (self.grid[r][c] == 'g') or (self.grid[r][c] == 'x')
        self.current_step +=1
        self.state = next_state

        return next_state, reward, done, {}

    def render(self):
        outfile = sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        r, c = self.state2coord[self.state]

        def ul(x):
            return "_" if x == " " else x

        if self.grid[r][c] == 'x':
            out[1 + r][2 * c + 1] = utils.colorize(out[1 + r][2 * c + 1], 'red', highlight=True)
        elif self.grid[r][c] == 'g':  # passenger in taxi
            out[1 + r][2 * c + 1] = utils.colorize(ul(out[1 + r][2 * c + 1]), 'green', highlight=True)
        else:
            out[1 + r][2 * c + 1] = utils.colorize(ul(out[1 + r][2 * c + 1]), 'yellow', highlight=True)

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(self.action_names[self.lastaction]))
        else:
            outfile.write("\n")
        outfile.flush()

    def render_policy(self, pol):
        outfile = sys.stdout
        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        r, c = self.state2coord[self.state]

        for s in range(self.Ns):
            r, c = self.state2coord[s]
            action = pol[s]
            # 'right', 'down', 'left', 'up'
            if action == 0:
                out[1 + r][2 * c + 1] = '>'
            elif action == 1:
                out[1 + r][2 * c + 1] = 'v'
            elif action == 2:
                out[1 + r][2 * c + 1] = '<'
            elif action == 3:
                out[1 + r][2 * c + 1] = '^'
            else:
                raise ValueError()

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

    def copy(self):
        new_env = GridWorldWithPits(grid=self.grid, txt_map=self.txt_map,
                                    proba_succ=self.proba_succ, uniform_trans_proba=self.uniform_trans_proba)
        return new_env

    def sample_transition(self, s, a):
        try:
            p = self.P[s, a]
        except:
            raise ValueError("Action {} cannot be executed in this state {}".format(action, self.state))
        next_state = np.random.choice(self.nb_states, 1, p=p).item()


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

    env.render()

    ### The states on the map are encoded as 
    # 0  1  2  3
    # 4  5  6  7
    # 8  9 10 11

    ### The actions on the map are encoded as 
    # 0 >
    # 1 v
    # 2 < 
    # 3 ^

