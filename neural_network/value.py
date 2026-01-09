class Value:





















    def __init__(self,data, _children=(), _op=''):
        self.data=data
        self._prev=set(_children) # Stores the "parents"
        self._op=_op # Stores the "operation " label
    def __add__(self, other):
        out=Value(self.data +other.data,(self,other), '+')
        return out
    def __mul__(self, other):
        out=Value(self.data *other.data, (self,other), '*')
        return out
    def __repr__(self):
        return f"Value(data={self.data})"

# 1. Create two 'Value' objects
a=Value(10)
b=Value(5)
c=a+b

print(c)