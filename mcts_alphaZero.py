# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode。动作->子节点的映射
        self._n_visits = 0
        self._Q = 0  # 收益预估
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # 策略网络输出的动作-概率元组
        for action, prob in action_priors:
            if action not in self._children:
                # 扩展的子节点的 parent 刚好是self
                # 初始化先验概率，其他属性都是0
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        # 扩展
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        # 选择置信收益最大的节点
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        # 更新
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.访问次数固定加1
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        # 更新收益的平均值, 整体来看 Q是 收益的平均值，而不是 获胜的次数这种 计数 概念。
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        # 回溯
        """Like a call to update(), but applied recursively for all ancestors."""
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            # 父子节点分属对立玩家，因此父节点的价值是子节点价值的相反数
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        # 计算置信收益
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        # 蒙特卡洛搜索树核心公式：获取当前阶段的价值收益预估(置信上界策略)，prior upper confidence for tree
        # 常数 * 先验概率(策略网络输出) * 父节点访问次数^0.5 / 当前节点的访问次数
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn  # 策略网络
        self._c_puct = c_puct  # 常数
        self._n_playout = n_playout  #

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        # 输入当前状态为state，输出子节点(对手)的策略分布，新的叶子节点价值(对手价值)
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            # node扩展(对手扩展)
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        # 基于子节点的价值取相反数 更新 当前node的价值
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # 通过执行指定次数的 MCTS 推演（playout），从当前游戏局面出发，计算出所有合法动作对应的选择概率，
        # 为AI落子提供依据
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        # [(动作, 节点访问次数)]
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        # 对访问次数取自然对数，作用是「平滑访问次数的差异」，避免高访问次数动作的优势过于极端，
        # 同时符合信息论中「概率与对数访问次数相关」的设定
        # 概率的本质：访问次数决定优先级

        # act_probs 的核心逻辑是「访问次数越多的动作，对应的概率越大」，
        # 因为 MCTS 的 _playout 过程中，更有价值的动作（胜率更高）会被反复选中，访问次数自然累积更多，这是 MCTS 决策的核心依据。
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            # 搜索树复用# 切换根节点
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            # 搜索树重置
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    # 基于游戏局面，结合MCTS搜索，最终输出一个具体的落子动作。
    def get_action(self, board, temp=1e-3, return_prob=0):
        # check 是否还有【空位】
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            # 模型训练
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                # 复用搜索树，将根节点切换为该动作对应的子节点
                self.mcts.update_with_move(move)

            # 实战博弈
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                # 重置搜索树，创建全新的根节点，放弃原有搜索结果
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
