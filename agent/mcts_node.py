# -*- coding: utf-8 -*-

import numpy as np


class Node(object):
    """A tree node in the Monte Carlo Tree Search.
       Each node keeps track of <Q, P, u>
       Q is its own value Q,
       P is prior probability P,
       u is its visit-count-adjusted prior score u.
    """

    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}

        self.n_visits = 0
        self.n_wins = 0

        self.q_value = 0  # Q-value

    def select(self, c=5):
        """ [Select phase] Return an action according to UCB value.

            :arg c: hyper-parameter to balance exploitation and exploration.
            :return <action, next_node> tuple """

        # Select a child-node according to UCB value.
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_ucb(c))

    def expand(self, action):
        """ [Expand phase] Create new child-node.
        :arg action: action to be expanded. """
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
        """ [Update phase] Update node values from leaf evaluation.
            :arg reward: the value of subtree evaluation from the current agent's perspective. """

        """ Update current node, including Q-value and #visit. """
        self.q_value = ((self.n_visits * self.q_value) + reward) / (self.n_visits + 1)
        self.n_visits += 1  # Count visit.
        if reward > 0:
            self.n_visits += 1

        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update(-reward)

    def get_ucb(self, c):
        """ Calculate the upper-confidence-bound (UCB) of some node.

            :arg: c hyper-parameter c is in (0, inf),
            UCB = [quality / times] + C * [sqrt(2 * ln(total_times) / times)]. """

        # Left part of UCB
        exploitation_score = self.q_value / (self.n_visits + 1)
        # Right part of UCB
        exploration_score = np.sqrt(2 * np.log(self.parent.n_visits) / (self.n_visits + 1))

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
