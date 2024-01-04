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


Broadcasting rules:

"""

try:
  import h5py
except:
  pass

import torch
t = torch.zeros(1,1)
tensor_manipulations = dict(
  # constructor
  constructors = [
    torch.ones,
    torch.zeros,
    torch.tensor,
    torch.randn,
    torch.rand,
    torch.from_numpy,
  ],
  
  copy = t.clone, # so aren't a view of the same storage
  to_numpy = t.numpy,
  
  # --- storage ---
  storage = t.untyped_storage,#()
  shape = t.shape,
  size = t.size,
  offset = t.storage_offset,
  stride = t.stride,
  check_continguous = t.is_contiguous,
  make_continguous = t.contiguous, # only if necessary
  
  move_to_gpu = [
    t.cuda,
    t.to,
  ],
  move_to_cpu = [
    t.to,
    t.cpu,
  ],
  
  dtype = t.dtype,
  change_dtype = [
    t.float,
    t.double,
    t.short,
    t.to(dtype=torch.float)
  ],
  
  
  # -- indexing/extraction
  one_element_tensor_to_element = t.item,
  transpose = [
    t.transpose,
    t.t,
    t.permute
  ],
  
  
  # -- views/dimensions
  add_singelton_dim = [
    t.unsqueeze,torch.unsqueeze
  ],
  remove_singleton_dim = [
    t.squeeze, torch.unsqueeze,
  ],
  reshape_no_copy = t.view,
  reshape_maybe_copy = t.reshape, #Contiguous inputs and inputs with compatible strides can be reshaped without copying
  
  
  # -- reductions
  max=torch.max,
  sum = torch.sum,
  mean = torch.mean,
  from_numpy = torch.from_numpy,
  
  
  #-- read/write
  load = torch.load,
  save = torch.save,
  
  # -- operations
  softmax = torch.nn.functional.softmax,
  sort=[torch.sort],
   
  # named dimensions,
  change_dim_name = t.refine_names,
  dim_names = t.names,
  
  #data types
  converting_device_dtype = t.to,
  
)

def check_if_tensors_same_storage(t1,t2):
  # old way that did not work
  #return id(t1.storage()) == id(t2.storage())
  
  return t1.untyped_storage().data_ptr() == t2.untyped_storage().data_ptr()

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


"""
when using torch.load and t.save, these
will save as file formats that only pytorch can interpret

For a more generic type can save as a hdf5 file

"""
hdf5_key_default = "tensor"
def save_hdf5(t,filepath,key=None):
  if key is None:
    key = hdf5_key_default
  
  f = h5py.File(filepath,'w')
  dset = f.create_dataset(
    key,
    data = t,
  )
  f.close()
  
def load_hdf5(filepath,key=None):
  if key is None:
    key = hdf5_key_default
  f = h5py.File(filepath, 'r')
  dset = torch.from_numpy(f[key][:])
  f.close()
  return dset
