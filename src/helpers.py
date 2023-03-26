import numpy as np
# return true if X was added to the set S
def set_add(s, x):
  return len(s) != (s.add(x) or len(s))

def percent(x, y, decimal = 2):
   return round(x / y * 100, decimal)
