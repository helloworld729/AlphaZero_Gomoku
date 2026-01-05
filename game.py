# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.last_move = None
        self.run_cnt=0  # 已经落了多少子
        self.current_player = None
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.already_move=[]

        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        # 一维坐标 到 玩家编号 的映射
        # do_move函数有具体置位过程
        # 作用：快速查询、胜负判断、状态记录
        self.states = {}

        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2
        print("Board:init: 初始化棋盘")

    def init_board(self, start_player=0):
        print("Board:init_board: 棋盘清零")
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        self.run_cnt=0  # 已经落了多少子
        self.already_move=[]

    def move_to_location(self, move):
        # 一维坐标 转化为 二维坐标
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    # 二维坐标 转化为 一维坐标
    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        # 初始化4类特征
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            # 一维坐标: 玩家编号
            moves, players = np.array(list(zip(*self.states.items())))
            # 当前选手的落子位置
            move_curr = moves[players == self.current_player]
            # 对手的落子位置
            move_oppo = moves[players != self.current_player]
            # 当前选手的落子特征
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            # 对手的落子特征
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # 对手 最后一步的落子特征
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        # 是否先手特征
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move, show=True):
        if show:
            print("当前选手={}, 执行动作={}, 并从棋盘合法动作中去除该动作，进行选手轮换，记录last_move".format(self.current_player, move))
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move
        self.run_cnt += 1
        self.already_move.append(int(move))


    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        print("Game:init: 初始化Game")

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    # 实战对弈
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            # 会在内部进行轮换
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    # 自我博弈训练
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        # 基于MCTS收集数据
        self.board.init_board()
        # 初始化选手
        print("Game:start_self_play: 获取棋盘2位选手", self.board.players)
        p1, p2 = self.board.players
        # 初始化状态、策略、选手
        states, mcts_probs, current_players = [], [], []
        while True:
            # MCTS 落子
            print("\n\nGame:start_self_play: 当前需要第{}次落子, 当前棋手={}, 已经落的子有:{}, 开始推演-->".format(
                self.board.run_cnt+1, self.board.current_player, self.board.already_move))
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # 保存: 状态、概率、选手
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # perform a move，do_move函数内部实现选手的轮换
            print("Game:start_self_play: 实际棋盘开始落子, 位置={}\n\n\n".format(move))
            self.board.do_move(move, show=False)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                print("Game:start_self_play: 对弈结束, 胜利的一方={}".format(winner))
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    # 基于 np.array(current_players) == winner 获取 胜利者 索引
                    # 由于 winners_z 和 current_players 数组长度相同，因此复用索引
                    # => 为自我对弈的每一步棋局状态，生成对应的「胜负价值标签」
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game:start_self_play: Game end. Winner is player:", winner)
                    else:
                        print("Game:start_self_play: Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
