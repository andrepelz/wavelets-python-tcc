import numpy as np

from typing import Callable
from numpy.typing import ArrayLike
from dataclasses import dataclass, field


def default_cost(data : ArrayLike) -> int:
    result = 0
    threshold = 1

    for sample in data:
        result += 1 if np.abs(sample) > threshold else 0

    return result


@dataclass
class TransformTree:
    @dataclass
    class TransformNode:
        data: ArrayLike = field(default=None)
        depth: int = field(default=0)
        parent: 'TransformTree.TransformNode' = field(default=None)
        left: 'TransformTree.TransformNode' = field(default=None)
        right: 'TransformTree.TransformNode' = field(default=None)
        cost: float = field(default=0.0)
        best_basis: bool = field(default=False)

        def _calculate_cost(self, cost_function: Callable) -> None:
            if self.left and self.right:
                current_cost = cost_function(self.data)
                children_cost = (
                    self.left.get_cost(cost_function) + 
                    self.right.get_cost(cost_function)
                )

                if current_cost <= children_cost:
                    self.mark_as_best_basis()
                    self.cost = current_cost
                else:
                    self.cost = children_cost
            else:
                self.mark_as_best_basis()
                self.cost = cost_function(self.data)

        def get_cost(self, cost_function: Callable) -> float:
            if self.cost == 0:
                self._calculate_cost(cost_function)
            
            return self.cost
        
        def mark_as_best_basis(self) -> None:
            def unmark_node(node : 'TransformTree.TransformNode') -> None:
                node.best_basis = False

                if node.left:
                    unmark_node(node.left)
                if node.right:
                    unmark_node(node.right)

            self.best_basis = True

            if self.left:
                unmark_node(self.left)
            if self.right:
                unmark_node(self.right)

        def __del__(self):
            if self.left:
                del self.left

            if self.right:
                del self.right

    root: TransformNode = field(default=None)
    total_depth: int = field(default=0)
    cost_function: Callable = field(default=default_cost)

    def __del__(self):
        if self.root:
            del self.root

    def add_node(self, data: ArrayLike) -> None:
        from collections import deque

        def _check_node(node: TransformTree.TransformNode, queue: deque) -> bool:
            if node:
                queue.append(node)

                return False
            else:
                return True
            
        if self.root:
            queue = deque()
            queue.append(self.root)

            while len(queue):
                curr_node = queue.popleft()
                curr_depth = curr_node.depth

                if _check_node(curr_node.left, queue):
                    curr_node.left = TransformTree.TransformNode(data=np.copy(data), depth=(curr_depth + 1), parent=curr_node)

                    if curr_depth == self.total_depth:
                        self.total_depth += 1

                    return
                
                if _check_node(curr_node.right, queue):
                    curr_node.right = TransformTree.TransformNode(data=np.copy(data), depth=(curr_depth + 1), parent=curr_node)

                    if curr_depth == self.total_depth:
                        self.total_depth += 1

                    return
        else:
            self.root = TransformTree.TransformNode(data=np.copy(data), depth=1)
            self.total_depth = 1

            return
    
    def get_full_transform(self) -> ArrayLike:
        from collections import deque

        result = np.array([], dtype=np.float64)

        if self.root:
            queue = deque()
            queue.append(self.root)

            while queue[0].depth < self.total_depth: # BFS approach to get last layer of the transform tree
                curr_node = queue.popleft()

                queue.append(curr_node.left)
                queue.append(curr_node.right)

            while len(queue):
                node = queue.popleft()
                result = np.append(result, node.data)

        return result

    def get_best_basis(self) -> ArrayLike:
        def build_stack(stack, node): # DFS approach to calculate the best basis
            stack.append(node)

            if node.right:
                build_stack(stack, node.right)
            if node.left:
                build_stack(stack, node.left)

        node = self.root
    
        node.get_cost(self.cost_function)

        from collections import deque

        result = np.array([], dtype=TransformTree.TransformNode)

        if self.root:
            stack = deque()
            
            build_stack(stack, self.root)

            while len(stack):
                node = stack.pop()

                if node.best_basis:
                    result = np.append(result, node)

        return result
    