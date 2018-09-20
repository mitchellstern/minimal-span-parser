""" generic A-Star path searching algorithm """
""" based on https://raw.githubusercontent.com/jrialland/python-astar/master/src/astar/__init__.py"""

from abc import ABCMeta, abstractmethod
from heapq import heappush, heappop

import time

class AStar:
    __metaclass__ = ABCMeta
    __slots__ = ()

    class SearchNode:
        __slots__ = ('data', 'fscore',
                     'closed', 'came_from', 'out_openset')

        def __init__(self, data, fscore=.0):
            self.data = data
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, other):
            return self.fscore > other.fscore

        def format_print(self, label):
            node_string = self.data.format_print(label)
            return '{} score: {}'.format(node_string, self.fscore)

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    @abstractmethod
    @abstractmethod
    def heuristic_cost(self, current, goal):
        """Computes the estimated (rough) distance between a node and the goal,
        this method must be implemented in a subclass. The second parameter is
        always the goal."""
        raise NotImplementedError

    @abstractmethod
    def real_cost(self, current):
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node):
        """For a given node, returns (or yields) the list of its neighbors.
        this method must be implemented in a subclass"""
        raise NotImplementedError

    @abstractmethod
    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        raise NotImplementedError

    @abstractmethod
    def move_to_closed(self, current):
        raise NotImplementedError

    # def reconstruct_path(self, last, reverse_path=False):
    #     def _gen():
    #         current = last
    #         while current:
    #             yield current.data
    #             current = current.came_from
    #     if reverse_path:
    #         return _gen()
    #     else:
    #         return reversed(list(_gen()))

        current_time = start_time = time.clock()
        cost_coeff = 1.
        searchNodes = AStar.SearchNodeDict()
        openSet = []
        goals = []
        # max_len = 0
        # max_node = None
        for strt in  start:
            if self.is_goal_reached(strt, goal):
                goals.append(strt)
            cost = self.real_cost(strt) + self.heuristic_cost(strt, goal, cost_coeff)
            startNode = searchNodes[strt] = AStar.SearchNode(strt, fscore=cost)
            heappush(openSet, startNode)
        while (time.clock() - start_time < time_out) and openSet and len(goals) < num_goals:
            current = heappop(openSet)
            if (time.clock() - current_time >= time_th):
                cost_coeff *= cost_coeff_rate
                for t in openSet:
                    t.fscore = self.real_cost(t.data) \
                            + self.heuristic_cost(t.data, goal, cost_coeff)
                current_time = time.clock()
            if verbose > 0:
                self.print_fn(current, 'current')
            if self.is_goal_reached(current.data, goal):
                # goals.append(self.reconstruct_path(current, reverse_path))
            current.out_openset = True
            current.closed = True
            self.move_to_closed(current.data)
            for neighbor in [searchNodes[n] for n in self.neighbors(current.data)]:
                if neighbor.closed:
                    continue
                neighbor.fscore = self.real_cost(neighbor.data) \
                                    + self.heuristic_cost(neighbor.data, goal, cost_coeff)
                neighbor.came_from = current

                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                if verbose > 1 :
                    self.print_fn(neighbor, 'neighbor')
        # return goals, max_node
        return goals
