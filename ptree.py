class Node:
    def __init__(self, key):
        self.key = key
        self.children = []

    def is_node_created_from_rule(self, rule):
        for i, child in enumerate(self.children):
            if rule.derivation[i] != child.key:
                return False
        return True

    def __repr__(self):
        if self.children:
            return '[{} {}]'.format(self.key, ' '.join(map(repr, self.children)))
        else:
            return self.key

class PTree:
    def __init__(self, root = None, probability = 0):
        self.root = root
        self.probability = probability

    def __repr__(self):
        return '({}): {}'.format(self.probability, repr(self.root))
