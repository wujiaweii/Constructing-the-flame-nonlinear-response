import numpy as np

def tiem(f):
    t=1/f
    return t*20

f=np.arange(10,1000,20)
t=0
for i in f:
    t=t+tiem(i)
print(t*10)



