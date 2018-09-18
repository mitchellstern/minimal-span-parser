from .astar import AStar
import trees

class NodeT(object):

    def __init__(self, left, right, rank, trees = []):

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

    def is_valid(self, miss_tag_any):
        assert isinstance(self.trees, list)
        assert len(self.trees) in [1,2]

        def helper(c_trees, comb_side, miss_side, miss_tag_any):

            assert isinstance(c_trees[0], trees.InternalMyParseNode)
            assert (c_trees[0].label in [trees.CR, trees.CL])
            assert len(c_trees[0].children) == 1

            for leaf in list(c_trees[1].leaves())[::-1]:
                # check that destination tree has missing leaves and
                # they combine to the proper side
                if isinstance(leaf, trees.MissMyParseNode) and leaf.label.startswith(miss_side):
                    if miss_tag_any:
                        return leaf
                    label = leaf.label.split(miss_side)[-1]
                    if isinstance(c_trees[0].children[-1], trees.InternalMyParseNode):
                        src_label = c_trees[0].children[-1].label
                    else:
                        src_label = c_trees[0].children[-1].tag
                    if src_label == label:
                        return leaf
            return None

        if len(self.trees) == 1:
            return True

        if all(isinstance(tree, trees.InternalMyParseNode) for tree in self.trees):
            #try combining left tree into right tree
            if self.trees[0].label == trees.CR and self.trees[0].is_no_missing_leaves():
                miss_node = helper(self.trees, trees.CR, trees.L, miss_tag_any)
                if miss_node is not None:
                    self.trees = [self.trees[1].combine_tree(self.trees[0], miss_node)]
                    return True

            #try combining right tree into left tree
            if isinstance(self.trees[1], trees.InternalMyParseNode):
                if self.trees[1].label == trees.CL and self.trees[1].is_no_missing_leaves():
                    miss_node = helper(self.trees[::-1], trees.CL, trees.R, miss_tag_any)
                    if miss_node is not None:
                        self.trees = [self.trees[0].combine_tree(self.trees[1], miss_node)]
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

    def __init__(self, ts_mat, no_val_gap):
        self.ts_mat = ts_mat
        self.miss_tag_any = no_val_gap
        self.cl = ClosedList()
        self.seen = []

    def is_max_len(self, current, max_len, max_node):
        if len(current.rank) > max_len:
            return len(current.rank), current
        else:
            return max_len, max_node

    def print_fn(self, current, name):
        print ('%s: range %s, rank %s, score %f'
                %(name, (current.data.left, current.data.right), current.data.rank, current.fscore))

    def heuristic_cost(self, current, goal, cost_coeff):
        r_rng = [0] if current.left == 0 else list(range(current.left))
        l_rng = list(range(current.right, goal.right))
        idx_range = r_rng + l_rng
        return cost_coeff * sum([self.ts_mat[rng][0][1] for rng in idx_range])

    def real_cost(self, current):
        # if current.is_valid(self.ts_mat[current.left][current.rank[0]]['tree'], self.miss_tag_any):
        idx_range = list(range(current.left, current.right))
        pos = zip(idx_range, current.rank)
        return sum([self.ts_mat[rng][rnk][1] for rng, rnk in pos])
        # return .0

    def move_to_closed(self, current):
        self.cl.put(current)

    def neighbors(self, node):
        neighbors = []
        for nb in self.cl.getl(node.right):
            nb_node = NodeT(node.left, nb.right, node.rank + nb.rank, node.trees + nb.trees)
            if nb_node not in self.seen and nb_node.is_valid(self.miss_tag_any):
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        for nb in self.cl.getr(node.left):
            nb_node = NodeT(nb.left, node.right, nb.rank + node.rank, nb.trees + node.trees)
            if nb_node.is_valid(self.miss_tag_any) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        if len(node.rank) == 1 and node.rank[0] + 1 < len(self.ts_mat[node.left]):
            rank = node.rank[0] + 1
            trees = [self.ts_mat[node.left][rank][0]]
            nb_node = NodeT(node.left, node.right, [rank], trees)
            if nb_node.is_valid(self.miss_tag_any) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        return neighbors

    def is_goal_reached(self, current, goal):
        if (current.left, current.right) == (goal.left, goal.right):
            if len(current.trees) == 1:
                return current.trees[0].is_no_missing_leaves()
        return False

def astar_search(beams, no_val_gap, num_goals, time_out,
                        time_th, cost_coeff_rate, verbose=1):

    n_words = len(beams)
    start = [NodeT(idx, idx+1, [0], [beams[idx][0][0]]) for idx in range(n_words)]
    goal = NodeT(0, n_words, [])
    # let's solve it
    path = Solver(beams, no_val_gap).astar(start, goal, num_goals, time_out,
                                            time_th, cost_coeff_rate)

    if path == []:
        return trees.LeafMyParseNode(0, '', '')
    path = list(path)[-1]
    return path.trees[0]
