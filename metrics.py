"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# PyTorch
from torch.nn import MSELoss, BCELoss
_mse_loss, _bce_loss = MSELoss(reduction="mean"), BCELoss(reduction="mean")

def MSLE(input, /, target):
	input, target = (input + 1).log(), (target + 1).log()
	return _mse_loss(input, target)

def BCE(input, /, target):
	return _bce_loss(input.float(), target.float())

def RMSE(input, /, target):
	return (target - input).square().mean().sqrt()
	
def Fscore(input, /, target, beta=1.0, dim=None, keepdim=False):
	# parse dimension(s)
	if dim is None:
		dims = list(range(target.ndim))
	elif isinstance(dim, int):
		dims = [dim]
	else:
		dims = [d for d in dim]
	dims = [range(target.ndim)[dim] for dim in dims]

	# compute F score
	true_positive = (target & input).sum(dim=dims, keepdim=keepdim)
	positive = input.sum(dim=dims, keepdim=keepdim)
	relevant = target.sum(dim=dims, keepdim=keepdim)
	b = beta**2
	return (1 + b) * true_positive / (positive + b * relevant)
