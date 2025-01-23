# Adapted from https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py

# The ‘sum-tree’ data structure used here is very similar in spirit to the array representation
# of a binary heap. However, instead of the usual heap property, the value of a parent node is
# the sum of its children. Leaf nodes store the transition priorities and the internal nodes are
# intermediate sums, with the parent node containing the sum over all priorities, p_total. This
# provides an efficient way of calculating the cumulative sum of priorities, allowing O(log N) updates
# and sampling. (Appendix B.2.1, Proportional prioritization)

import numpy as np

class SumTree:
    def __init__(self, capacity):
        """
        SumTree for storing priorities and sampling efficiently.
        capacity: the maximum number of transitions storable.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

        self.write = 0 # next write position
        self.size = 0 # current size of the tree

    @property
    def total(self):
        return self.tree[0]

    def update(self, data_idx, value):
        idx = data_idx + self.capacity - 1  # child index in tree array
        change = value - self.tree[idx]

        self.tree[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def add(self, priority, data):
        self.data[self.write] = data
        self.update(self.write, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.tree):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.tree[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.tree[left]

        data_idx = idx - self.capacity + 1

        return data_idx, self.tree[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.tree.__repr__()}, data={self.data.__repr__()})"