# -*- coding: utf-8 -*-

import numpy as np

from env.gomoku import Gomoku
from agent.meta_player import MetaPlayer

from agent.mcts_agent import MCTSMetaPlayer


class RandomAgent(MetaPlayer):

    def __init__(self):
        super(RandomAgent, self).__init__()

    def set_player_index(self, p):
        pass

    def select_action(self, state):
        valid_action_list = state["action"]
        if valid_action_list:
            action = np.random.choice(valid_action_list)
            print("[Rand %s]" % action, end=" ")
            return action
        else:
            print("Game board is full.")
            return -1

    def reset_player(self):
        pass

