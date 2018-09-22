from itertools import chain
from .astar import AStar
import trees
import numpy as np

class AstarNode(object):

    def __init__(self, left, right, rank, trees=[]):

        assert isinstance(left, int)
        self.left = left

        assert isinstance(right, int)
        self.right = right

        assert isinstance(rank, list)
        self.rank = rank

        assert isinstance(trees, list)
        self.trees = trees

    def __eq__(self, other):
        return self.rank == other.rank and (self.left, self.right) == (other.left, other.right)

    def __hash__(self):
        return id(self)

    def format_print(self, label):
        pair = '({},{})'.format(self.left, self.right)

        ranks_split = np.split(np.array(self.rank), np.where(np.diff(self.rank))[0] + 1)
        ranks = ', '.join(['{{{}}}{}'.format(r[0], len(r)) for r in ranks_split])

        MY_LENGTH_CONSTRAINT = len(ranks_split) * 7
        node_string = '[{}:] node: {: <8} rank: [{: <{mlc}}]'.format(label, pair, ranks,
                                                        mlc = MY_LENGTH_CONSTRAINT)

        for i, tree in enumerate(self.trees):
            pair = '({},{})'.format(tree.left, tree.right)
            # ptb = tree.convert().linearize()
            node_string = '{} tree[{}]: {: <8}'.format(node_string, i, pair)

        return node_string

    def is_valid(self, keep_valence_value):
        assert isinstance(self.trees, list)
        assert len(self.trees) == 2

        def helper(_trees, comb_side, miss_side):

            assert isinstance(_trees[0], trees.InternalMyParseNode)
            assert (_trees[0].label in [trees.CR, trees.CL])
            assert len(_trees[0].children) == 1
            #TODO fix combination order --> incorrect order
            leaves = []
            label = _trees[0].children[-1].bracket_label()
            for leaf in _trees[1].missing_leaves():
                if leaf.label.startswith(miss_side):
                    missing_label = leaf.label.split(miss_side)[-1]
                    if not keep_valence_value:
                        leaves.append(leaf)
                    elif missing_label == label:
                        leaves.append(leaf)
            return leaves

        if all(isinstance(tree, trees.InternalMyParseNode) for tree in self.trees):
            #Trying to combine Left Tree --> Right Tree
            if self.trees[0].label == trees.CR and len(list(self.trees[0].missing_leaves()))==0:
                leaves = helper(self.trees, trees.CR, trees.L)
                if leaves != []:
                    self.trees = [self.trees[1].combine(self.trees[0].children[0], leaves[-1])]
                    return True

            #Trying to combine Right Tree --> Left Tree
            if self.trees[1].label == trees.CL and len(list(self.trees[1].missing_leaves()))==0:
                leaves = helper(self.trees[::-1], trees.CL, trees.R)
                if leaves != []:
                    self.trees = [self.trees[0].combine(self.trees[1].children[0], leaves[0])]
                    return True
        return False


class ClosedList(object):

    def __init__(self):
        self.lindex = {}
        self.rindex = {}

    def put(self, node):
        if node.left in self.lindex:
            if node not in self.lindex[node.left]:
                self.lindex[node.left].append(node)
        else:
            self.lindex[node.left] = [node]

        if node.right in self.rindex:
            if node not in self.rindex[node.right]:
                self.rindex[node.right].append(node)
        else:
            self.rindex[node.right] = [node]

    def getr(self, idx):
        return self.rindex.get(idx, [])

    def getl(self, idx):
        return self.lindex.get(idx, [])


class Solver(AStar):

    def __init__(self, grid, keep_valence_value):
        self.grid = grid
        self.keep_valence_value = keep_valence_value
        self.cl = ClosedList()
        self.seen = []

    def heuristic_cost(self, node, goal, cost_coefficient):
        left = list(range(node.left))
        right = list(range(node.right, goal.right))
        return cost_coefficient * sum([self.grid[i][0][1] for i in chain(left, right)])

    def real_cost(self, node):
        position = zip(range(node.left, node.right), node.rank)
        return sum([self.grid[i][rank][1] for i, rank in position])

    def fscore(self, node, goal, cost_coefficient):
        real_cost = self.real_cost(node)
        heuristic_cost = self.heuristic_cost(node, goal, cost_coefficient)
        return real_cost + heuristic_cost

    def move_to_closed(self, node):
        self.cl.put(node)

    def neighbors(self, node):
        neighbors = []
        for nb in self.cl.getl(node.right):
            nb_node = AstarNode(node.left, nb.right, node.rank + nb.rank, node.trees + nb.trees)
            if nb_node not in self.seen and nb_node.is_valid(self.keep_valence_value):
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        for nb in self.cl.getr(node.left):
            nb_node = AstarNode(nb.left, node.right, nb.rank + node.rank, nb.trees + node.trees)
            if nb_node not in self.seen and nb_node.is_valid(self.keep_valence_value):
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        if len(node.rank) == 1 and node.rank[0] + 1 < len(self.grid[node.left]):
            rank = node.rank[0] + 1
            nb_node = AstarNode(node.left, node.right, [rank], [self.grid[node.left][rank][0]])
            if nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        return neighbors

    def is_goal_reached(self, node, goal):
        if (node.left, node.right) == (goal.left, goal.right):
            if len(node.trees) == 1:
                return len(list(node.trees[0].missing_leaves()))==0
        return False

def astar_search(grid, keep_valence_value, astar_parms, verbose=0):

    n_words = len(grid)
    start = [AstarNode(idx, idx+1, [0], [grid[idx][0][0]]) for idx in range(n_words)]
    goal = AstarNode(0, n_words, [])
    # let's solve it
    nodes = Solver(grid, keep_valence_value).astar(start, goal, *astar_parms, verbose)

    return nodes
    # if nodes == []:
    #      # return trees.LeafMyParseNode(0, '', '')
    #      return nodes
    # return nodes[0].trees[0]
