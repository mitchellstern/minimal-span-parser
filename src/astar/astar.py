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
    def heuristic_cost(self, node, goal):
        """Computes the estimated (rough) distance between a node and the goal,
        this method must be implemented in a subclass. The second parameter is
        always the goal."""
        raise NotImplementedError

    @abstractmethod
    def real_cost(self, node):
        raise NotImplementedError

    @abstractmethod
    def fscore(self, node, goal):
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

    def astar(self, start, goal, n_goals, time_out, n_cost_reductions, cost_reduction_rate, verbose):

        cost_coefficient = 1.
        searchNodes = AStar.SearchNodeDict()
        openSet = []
        goals = []
        for strt in  start:
            if self.is_goal_reached(strt, goal):
                goals.append(strt)
            cost = self.fscore(strt, goal, cost_coefficient)
            startNode = searchNodes[strt] = AStar.SearchNode(strt, fscore=cost)
            heappush(openSet, startNode)

        total_time_out = time_out * n_cost_reductions
        current_time = start_time = time.clock()
        while (time.clock() - start_time < total_time_out) and openSet and len(goals) < int(n_goals):
            current = heappop(openSet)
            if (time.clock() - current_time >= time_out):
                cost_coefficient *= cost_reduction_rate
                for t in openSet:
                    t.fscore = self.fscore(t.data, goal, cost_coefficient)
                current_time = time.clock()
            if verbose > 0: print(current.format_print('current'))

            if self.is_goal_reached(current.data, goal):
                goals.append(current.data)
            current.out_openset = True
            current.closed = True
            self.move_to_closed(current.data)
            for neighbor in [searchNodes[n] for n in self.neighbors(current.data)]:
                if neighbor.closed:
                    continue
                neighbor.fscore = self.fscore(neighbor.data, goal, cost_coefficient)
                neighbor.came_from = current

                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)

                if verbose > 1: print(neighbor.print_fn('neighbor'))

        return goals
