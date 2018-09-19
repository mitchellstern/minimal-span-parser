import collections.abc

R = '}'
L = '{'
CR = '>'
CL = '<'
ANY = '*'

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

        for child in self.children:
            child.parent = self

        self.parent = None

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

    def myconvert(self, dependancy, index=0):
        tree = self
        children = []
        for child in tree.children:
            children.append(child.myconvert(dependancy, index=index))
            index = children[-1].right

        return InternalMyParseNode(tree.label, children)


class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

        self.parent = None

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

    def myconvert(self, dependancy, index=0):
        return LeafMyParseNode(index, self.tag, self.word)(dependancy[index] - 1)


class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)


class MyParseNode(object):
    pass

class InternalMyParseNode(MyParseNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, MyParseNode) for child in children)
        assert children
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        for child in self.children:
            child.parent = self

        self.left = children[0].left
        self.right = children[-1].right

        self.parent = None

    def __call__(self, keep_valence_value):
        self.collapse(keep_valence_value)
        return self

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label, children)
        return tree

    def collapse(self, keep_valence_value):

        def helper(current, sibling):
            side = L if current.left > sibling.left else R
            if not keep_valence_value:
                return side+ANY
            elif isinstance(sibling, LeafMyParseNode):
                return side+sibling.tag
            else:
                return side+sibling.label

        # Recursion
        flag = CR
        for child in self.children:
            winner_child_leaf = child.collapse(keep_valence_value)

            # Reached end of path can add flag
            if winner_child_leaf.dependancy in range(self.left, self.right) or (flag == CL):
                winner_child_leaf.label + (flag,)
            else:
                # only single child will move to parent
                # and its value will be the one that is returned
                # to the parent
                winner_child_leaf.label + (self.label,)
                winner_child_leaf.label+ tuple([helper(child, sibling) for sibling in child.siblings()])
                ret_leaf_node = winner_child_leaf

                # once we reached here, it means that
                # this path includes the parent and thus flag
                # direction should flip
                flag = CL

        return ret_leaf_node

    def siblings(self):
        return [child for child in self.parent.children if child != self]

    def is_no_missing_leaves(self):
        for leaf in self.leaves():
            if isinstance(leaf, MissMyParseNode):
                return False
        return True

    def combine_tree(self, node_to_merge, node_to_remove):
        assert isinstance(node_to_merge, InternalMyParseNode)
        assert (node_to_merge.label in [CL, CR])
        assert len(node_to_merge.children) == 1
        assert isinstance(node_to_remove, MissMyParseNode)
        assert (node_to_remove in self.leaves())

        node_to_merge = node_to_merge.children[0]
        node_to_merge.parent = node_to_remove.parent
        children = node_to_remove.siblings() + [node_to_merge]
        children = sorted(children, key = lambda child: child.left)
        node_to_remove.parent.children = children

        return self

class LeafMyParseNode(MyParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def __call__(self, dependency):
        assert isinstance(dependency, int)
        self.dependancy = dependency
        return self

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def siblings(self):
        return [child for child in self.parent.children if child != self]

    def collapse(self, keep_valence_value):

        self.label = tuple(self.tag)
        return self


class MissMyParseNode(MyParseNode):
    def __init__(self, label, index = 0):
        self.label = label
        self.left = index
        self.right = index + 1

    def leaves(self):
        yield self

    def siblings(self):
        return [child for child in self.parent.children if child != self]

    def convert(self):
        return LeafTreebankNode(self.label, self.label)

def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":

            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1
        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]
                trees[i].parent = None

    return trees
