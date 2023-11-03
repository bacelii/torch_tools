"""
Torch functions:

t.sum(index)
t.max(index)

--- indexing ---
... (elipses): means insert as many dimensions so 
            that slicing happens over all dimensions
why? so that you don't have to know how many total dimensions
there are (unknown leading or trailing)

for 4d tensor
x[...,0] is same as x[:,:,:,0]
x[0,...] is same as x[0,:,:,:]


Syntax rules: 
1) if there is an _ at end of function name, it is an inplace function
  (aka it does not return an object so don't need to restore it)


"""

import torch
t = torch.zeros(1)
tensor_manipulations = dict(
  
  # -- indexing/extraction
  one_element_tensor_to_element = t.item(),
  
  # -- views/dimensions
  add_singelton_dim = [
    t.unsqueeze,torch.unsqueeze
  ],
  
  # -- reductions
  max=torch.max,
  
  #-- read/write
  load = torch.load,
  
  # -- operations
  softmax = torch.nn.functional.softmax,
  sort=[torch.sort],
  
)


def align_named_tensors(t1,t2):
  """
  Purpose: Assuming t1 and t2 are tensors with named dimensions.
  Will return tensor 1 with the follow changes:
  1) all maching named dimensions are reshuffled to align with t2
  2) any missing dimensions will have a size 1 dummy dimension inserted
  """
  return t1.aligned_as(t2)


def scalar_from_tensor(t):
  """
  Purpose: Will convert a tensor that just a has
  one element to extracting that element
  
  Example:
  t = torch.tensor([10])
  t.item()
  """
  return t.item()
