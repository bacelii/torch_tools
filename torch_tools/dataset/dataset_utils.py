"""
Provide import wrappers and information for
implementing custom datasets

Notes
----

When definig a dataset need to define 3 things for subclass
1) init
2) __len__
3) __getitem__:
    IMPORTANT: need to raise an IndexError if the index requested is larger than length
    (or else will loop forever)
"""

import torch

def example_get_item_with_error_raised(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        X and Y have the same shape, but Y is shifted left 1 position

        Pseudocode: 
        1) Gets the string starting and ending index (based on sample index and sequence length)
        2) Generate X as slice of self.X and y as +1 shifted slice of self.X
        3) Converts the lists to tensors (MAIN JOB OF A DATASET)
        """
        if index >= len(self):
            raise IndexError
            
        start_idx = index * self.seq_length
        end_idx = (index + 1) * self.seq_length

        print(f"start_idx = {start_idx}, end_idx = {end_idx}")

        X = torch.tensor(self.X[start_idx:end_idx]).float()
        y = torch.tensor(self.X[start_idx+1:end_idx+1]).float()
        return X, y

