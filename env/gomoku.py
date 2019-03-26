# -*- coding: utf-8 -*-
__author__ = "vonLynnShawn"
__copyright__ = "Copyright (C) 2019 vonLynnShawn"
__license__ = "MIT LICENSE"
__version__ = "1.1"

from env.game import Game
import numpy as np


class Gomoku(Game):

    def __init__(self, board_size=10, num4win=5):
        super(Gomoku, self).__init__()

        """ Init board information """
        self.board_size = board_size
        self.num4win = num4win
        if self.board_size < self.num4win:
            raise Exception('Board width and height can not be '
                            'less than {}'.format(self.num4win))

        # Initial board state to 0.
        self.board_state = np.full((self.board_size, self.board_size), 0, dtype=int)

        """ Init player information """
        self.players_dict = {
            -1: None,
            +1: None,
        }
        self.current_player_id = -1

        """ Init state-action information """
        self.available_action_space = list(range(self.board_size ** 2))
        self.invalid_action_space = []
        self.last_move = -1  # Use to determinate the terminal.

    def reset(self):
        """ Reset game variable """
        self.board_state = np.full((self.board_size, self.board_size), 0, dtype=int)
        self.available_action_space = list(range(self.board_size ** 2))
        self.invalid_action_space = []
        self.last_move = -1  # Use to determinate the terminal.

        return {"state": self.board_state,
                "action": self.available_action_space,
                "reward": 0,
                "terminal": 0,
                "last_move": -1}

    def step(self, action):
        """ Execute player's action, return new state.
            :arg action: integer
                Action that player chose.
            :return new_state, is_terminal
                new_state includes board state, valid/invalid action space. """

        """ Check validation of action. """
        if not self.available_action_space:
            return {"terminal": True, "reward": 0}

        if not self._is_valid_move(action):
            # Choose a valid action randomly.
            action = np.random.choice(self.available_action_space)

        """ Execute current action. """
        x, y = self._int_to_coordinate(action)
        self.board_state[x, y] = self.current_player_id

        """ Determinate whether reach the terminal. """
        self.last_move = action
        terminal_info, reward = self._is_terminal()

        """ Update action space and player information. """
        self.invalid_action_space.append(action)
        self.available_action_space.remove(action)

        """ Switch player """
        self.current_player_id = 1 if self.current_player_id == -1 else -1

        new_state = {
            "state": self.board_state,
            "action": self.available_action_space,
            "reward": reward,
            "terminal": terminal_info,
            "last_move": self.last_move
        }

        return new_state

    def run(self, player_a, player_b, is_show=True):
        """ Play the game for two players.
        :arg player_a: player object, the first player.
        :arg player_b: player object, the second player.
        :arg is_show: bool, whether to illustrate the game board in the end.
        """

        self.players_dict[-1] = player_a
        self.players_dict[+1] = player_b

        state = self.reset()

        # Choose a first-hand player randomly.
        self.current_player_id = np.random.choice([-1, 1])

        while True:
            # Players select and execute their actions in turn.
            action = self.players_dict[self.current_player_id].select_action(state)
            state = self.step(action)

            if state["terminal"]:  # Check whether the game is over.
                break

        # Illustrate game board in the end.
        if is_show:
            self.visualize()

        return state["reward"]

    def visualize(self):
        """ Illustrate the game board after the game is over. """

        print("The latest move is: ", self._int_to_coordinate(self.last_move))
        if self._last_player_id() == -1:
            print("O wins")
        elif self._last_player_id() == +1:
            print("X wins")
        else:
            print("Tie")

        board_list = list(self.board_state)
        print("-" * len(board_list))
        for i in range(len(board_list)):
            print("|", end=" ")
            for j in range(len(board_list)):
                if board_list[i][j] == 0:
                    print(" ", end=" ")
                elif board_list[i][j] == +1:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print("|")
        print("-" * len(board_list))

    def _is_terminal(self):
        """ Check whether current episode reaches terminal.

            :return tuple<is_terminal, reward>,
                is_terminal(bool) indicate whether game is over.
                reward(int) indicate the winner's id
                    -1 : Player A wins
                     0 : Tie
                    +1 : Player B wins
        """

        board_size = self.board_size
        num4win = self.num4win
        player_id = self.current_player_id

        y, x = self._int_to_coordinate(self.last_move)

        # 1. Check vertical direction
        begin_y_index = max(0, y-num4win+1)
        end_y_index = min(y+num4win, board_size)
        terminal_trigger = 0
        for i in range(begin_y_index, end_y_index):  # y-axis
            if self.board_state[i, x] == player_id:
                terminal_trigger += 1
            else:
                terminal_trigger = 0

            if terminal_trigger == num4win:
                return True, player_id

        # 2. Check horizontal direction
        begin_x_index = max(0, x-num4win+1)
        end_x_index = min(x+num4win, board_size)
        terminal_trigger = 0
        for i in range(begin_x_index, end_x_index):  # x-axis
            if self.board_state[y, i] == player_id:
                terminal_trigger += 1
            else:
                terminal_trigger = 0

            if terminal_trigger == num4win:
                return True, player_id

        # 3. Check Left-Top to Right-Button
        begin_margin = min(min(x, y), num4win-1)
        end_margin = min(min(board_size-x-1, board_size-y-1), num4win-1)
        oblique_range = begin_margin + end_margin + 1

        if oblique_range >= num4win:
            terminal_trigger = 0
            for i in range(oblique_range):
                if self.board_state[y-begin_margin+i, x-begin_margin+i] == player_id:
                    terminal_trigger += 1
                else:
                    terminal_trigger = 0

                if terminal_trigger == num4win:
                    return True, player_id

        # 4. Check Right-Top to Left-Button
        begin_margin = min(min(x, board_size-y-1), num4win-1)
        end_margin = min(min(board_size-x-1, y), num4win-1)
        oblique_range = begin_margin + end_margin + 1

        if oblique_range >= num4win:
            terminal_trigger = 0
            for i in range(oblique_range):
                if self.board_state[y+begin_margin-i, x-begin_margin+i] == player_id:
                    terminal_trigger += 1
                else:
                    terminal_trigger = 0

                if terminal_trigger == num4win:
                    return True, player_id

        " Board is full but nobody wins. "
        if len(self.available_action_space) == 0:
            return True, 0  # Tie
        else:
            return False, 0  # Continue the game.

    def _last_player_id(self):
        """ Return the id of player who move latest. """
        return -1 if self.current_player_id == 1 else 1

    def _int_to_coordinate(self, action):
        """ Convert move (integer) to 2-D coordinate. For example, 3*3 board.
                6 7 8
                3 4 5
                0 1 2
        """
        x = action // self.board_size
        y = action % self.board_size
        return [x, y]

    def _coordinate_to_int(self, coordinate):
        """ Convert 2d coordinate to int. """

        if len(coordinate) != 2:  # Check the validation of input coordinate.
            return -1

        x, y = coordinate
        move_int = x * self.board_size + y

        if move_int >= self.board_size ** 2:
            return -1
        else:
            return move_int

    def _is_valid_move(self, action):
        """ Check whether current move is valid. """
        return action in self.available_action_space


if __name__ == '__main__':
    "Example Code"
    game = Gomoku()
    # player1 = SomePlayer()
    # player2 = AnotherPlayer()
    # gomoku.run(player1, player2)
