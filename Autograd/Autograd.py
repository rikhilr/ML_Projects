import math

def dfs(startingNode):
    visited = set()
    result = []

    def visit(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                visit(child)
            result.append(node)
    
    visit(startingNode)
    return result

class Value:
    def __init__(self, value, label = '', _children=(), _op=''):
        self.grad = 0
        self.data = value
        self.label = label
        
        self._op = _op
        self._prev = set(_children)
        self._backward = lambda : None
    
    def backward(self):
        self.grad = 1.0
        for node in reversed(dfs(self)):
            node._backward()

    def tanh(self):
        out = Value(math.tanh(self.data), _children = (self, ), _op='tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value({self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children = (self, other), _op = '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self * -1

    def __pow__(self, exponent):
        out = Value(self.data ** exponent, _children=(self,), _op=f'^{exponent}')

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children = (self, other), _op = '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self - other
