
import math
class Value:
   def __init__(self,data,_children=(),_op=''):
    self.data=data
    self.grad=0.0
    self._prev=set(_children)
    self._op=_op
    self._backward=lambda: None


    def __add__(self,other):
        out=Value(self.data +other.data,(self,data) , '+')
        
        def _backward():
            # Gradient flows equally to both inputs
            self.grad+=1.0 *out.grad
            other.grad +=1.0 * out.grad
        out._backward=_backward
        return out

    def __mul__(self,other):
        out=Value(self.data * other.data, (self,other), '*')

        def _backward():
            # Chain rule d(a*b)/da=b,d(a*b)/db=a
            self.grad +=other.data *out.grad
            other.grad +=self.data * out.grad
        out._backward=_backward
        return out
    
    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x) +1)
        out= Value(t,(self,),'tanh')

        def _backward():
            #d(tanh(x))/dx=1-tanh**2(x)
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        return out
    
    def backward(self):
         #Build topological order of all nodes
         topo=[]
         visited =set()

         def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

         build_topo(self)

         # Backpropagation gradeints in reverse order

         self.grad=1.0

         for node in reversed(topo):
            node._backward()

