# -*- coding: utf-8 -*-
""" A pure implementation of the Monte Carlo Tree Search. """

import numpy as np
import copy

from agent import meta_player
from agent.mcts_node import Node


class MCTSMetaPlayer(meta_player.MetaPlayer):
    """ Monte Carlo Tree Search has 4 main steps:
        Selection Phase, Expansion Phase, Simulation Phase, Update Phase.
    """

    def __init__(self, env, c=5, train_episode=1500):
        """ :param env: game environment
            :param c: hyper-parameter of UCB exploration.
            :param train_episode: number of simulation.
        """
        super(MCTSMetaPlayer, self).__init__()
        self.player = None

        self.c = c  # hyper-parameter of UCB exploration.

        self.root = Node(None)
        self.env = copy.deepcopy(env)
        self.train_episode = train_episode
        self.last_move = -1

        self.train_mode = True

    def set_player_index(self, p):
        self.player = p

    def reset_player(self):
        self._update_current_node(-1)

    def select_action(self, state):
        """ Select action according to current state.
             :arg state: dict, complete game information
                ['state']: board state,
                ['action']: available action lists,
                ['terminal']: bool, whether reach terminal,
                ['reward']: int, complete reward in the game.
                ['last_move']: int,last move in current episode.
            :return: selected action. """

        available_action_list = state["action"]

        if len(available_action_list) == 0:
            print("WARNING: The board is full.")
            return -1

        if state["terminal"]:
            print("Reach Terminal.")
            return -1

        # Update the child node corresponding to the opponent's action to the new root node.
        self._clone(state)
        self._update_current_node(state["last_move"])

        if self.train_mode:  # Training
            for _ in range(self.train_episode):
                self._simulate(state=state, env=copy.deepcopy(self.env))

        """ Select the most frequent visited action of current node. """
        if not self.root.children:
            raise ValueError("Current node has no sub-node. ")

        # Sort sub-node according to visited num.
        sorted_node_list = sorted(self.root.children.items(),
                                  key=lambda sub_node: (sub_node[1].visit_num,
                                                        sub_node[1].q_value))
        self.last_move = sorted_node_list[-1][0]

        # Update the child node corresponding to the played action to the new root node.
        self._update_current_node(self.last_move)

        return self.last_move

    def _update_current_node(self, last_move):
        """ Update current tree node to its child node according to the lasted move.
            The child node corresponding to the played action becomes the new root node.
            The subtree below this child is retained along with all statistic,
            while remainder of the tree is discarded.

            :arg last_move: int, the latest move in current episode. """

        # Initialize root node.
        if last_move == -1:
            self.root = Node(None)
            return

        # Update root node.
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None)  # Create a new child node.

    def _simulate(self, state, env):
        """ Run a single episode from the root to the leaf,
            getting a value at the leaf and propagating it back through its parents.

            :arg state: the complete game state in current time step,
                        including board state, valid/invalid action space, etc.
            :arg env: copied game environment.
        """
        node = self.root

        """ [Selection Phase] 
             Starting at the root node, a tree policy based on the action values
             attached to the edges of the tree traverses the tree to select a leaf node.
        """
        while True:
            if not node.is_leaf() and node.is_full_expanded(len(state['action'])):
                action, node = node.select(self.c)  # Greedy select next move.
                state = env.step(action)  # Execute action
            else:
                break  # Start expanding.

        """ [Expansion Phase]
            On some iterations the tree is expanded from the selected leaf node by 
            adding one or more child nodes reached from the selected node via unexplored actions.
        """
        if len(state["action"]) == 0:
            return
        else:
            action = np.random.choice(state["action"])

        node.expand(action)
        node = node.children[action]
        state = env.step(action)

        """ [Play-out Phase] 
            From the selected node, or from one of its newly-expanded child nodes (if any), 
            simulation of a complete episode is run with actions selected by the play-out policy. 
            
            The result is a Monte Carlo trial with actions selected first by 
            the tree policy and beyond the tree by the play-out policy.
        """
        reward = node.play_out(state, env=env)

        """ [Update Phase] 
            The return generated by the simulated episode is backed up to update, or to initialize, 
            the action values attached to the edges of the tree traversed by the tree policy in this
            iteration of MCTS. No values are saved for the states and actions visited by the roll-out
            policy beyond the tree. 
        """
        node.update(reward)

    def _clone(self, state):
        self.env.board_state = state['state']
        self.env.available_action_space = state['action']
        self.env.last_move = state['last_move']

    def __str__(self):
        return "Monte Carlo Tree Search {}".format(self.player)
