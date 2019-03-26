# -*- coding: utf-8 -*-


class MetaPlayer:

    def __init__(self):
        pass

    def select_action(self, state):
        raise NotImplementedError

    def set_player_index(self, p):
        raise NotImplementedError

    def reset_player(self):
        raise NotImplementedError
