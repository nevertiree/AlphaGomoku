# -*- coding: utf-8 -*-

import numpy as np


class Node(object):
    """A tree node in the Monte Carlo Tree Search. """

    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}  # child nodes

        self.visit_num = 0  # N(s,a) is the visit count
        self.w_value = 0  # W(s,a) is the total action-value
        self.q_value = 0  # Q(s,a) is the mean action-value (Q-value)

        # Note that P(s,a) is used in AlphaGoZero, not in pure Monte Carlo Tree Search)
        self.prob = 1  # P(s,a) is the prior probability of selecting this node

    def select(self, c=5):
        """ [Select phase]
            Select action according to the statistics in the search tree,

            a = arg_max_{a} Q(s_t,a)+ U(s_t,a)
            where U(s_t,a) uses UCT algorithm or P_UCT algorithm.

            :arg c: constant, a hyper-parameter determine the level of exploration.
            :return <action, next_node> tuple """

        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_ucb(c))

    def expand(self, action, is_full_expand=False):
        """ [Expand phase]
            Expand leaf node and initialize child nodes.

            :arg action: int or list
                Action(s) to be expanded.
            :arg is_full_expand: bool
                Determine using full expansion or dynamic expansion.
        """
        if is_full_expand:  # Full Expansion
            if isinstance(action, list):
                raise TypeError("Full expansion need an action list, but got ", type(action))
            for a in action:
                self.children[a] = Node(self)
        else:  # Dynamic Expansion
            if isinstance(action, int):
                raise TypeError("Full expansion need an integer action, but got ", type(action))
            self.children[action] = Node(self)

    @staticmethod
    def play_out(state, env, limit=1000):
        """ [Play-out phase] Simulation util the terminal.

            Use the random play-out policy to play until the end of the game,
            returning +1 if the current agent wins, -1
            if the opponent wins, and 0 if it is a tie.

            :arg state: dict
                Complete game information, include:
                ['state']: np.array, board state,
                ['action']: list, available action lists,
                ['terminal']: bool, whether reach terminal
                ['reward']: int, complete reward in the game,
                ['last_move']: int, last move.
            :arg env: Game environment
            :arg limit: int, computational budget limit.
            :return leaf_node """

        count = 1

        for _ in range(limit):

            if state["terminal"]:  # Game is terminal.
                break
            if not state["action"]:  # Game board is full.
                break

            action = np.random.choice(state["action"])
            state = env.step(action)
            count += 1
        else:
            print("WARNING: Roll-out reached move limit")

        return ((-1) ** (count+1)) * state["reward"]

    def update(self, reward):
        """ [Update phase]
            The node statistic are updated in a backward pass through each step t.
            :arg reward: the value of subtree evaluation from the current agent's perspective.
        """

        self.visit_num += 1  # N(s_t,a_t)+1
        self.w_value += reward  # W(s_t,a_t)+v
        self.q_value = self.w_value / self.visit_num

        if self.parent:
            self.parent.update(-reward)

    def get_ucb(self, c):
        """ Calculate the upper-confidence-bound (UCB) of some node.
            :arg: c hyper-parameter c is in (0, inf),
            :return UCB value
            UCB = [quality / times] + C * [sqrt(2 * ln(total_times) / times)].
        """

        # Left part of UCB
        exploitation_score = self.q_value / (self.visit_num + 1)
        # Right part of UCB
        exploration_score = np.sqrt(2 * np.log(self.parent.visit_num) / (self.visit_num + 1))

        return exploitation_score + c * exploration_score

    def is_leaf(self):
        """ Check whether current node is a leaf.
            (i.e. no nodes below this have been expanded). """
        return self.children == {}

    def is_full_expanded(self, full_expanded_num):
        """ Check whether current node is full expanded. """
        return not len(self.children) < full_expanded_num

    def is_root(self):
        return self.parent is None
