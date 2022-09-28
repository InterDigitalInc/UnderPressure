"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# Python
import math, time

# Misc
import torch

## Tensor helpers
def parse_dim(dim: int, ndim: int) -> int:
	if not isinstance(ndim, int):
		raise TypeError("Expected '{}' but got '{}'.".format(int, type(ndim)))
	if not ndim >= 0:
		raise ValueError("Number of dimensions must be ≥ 0 but got {}.".format(ndim))
	if not isinstance(dim, int):
		raise TypeError("Expected '{}' but got '{}'.".format(int, type(ndim)))
	if dim < -ndim or dim >= ndim:
		raise IndexError("Dimension {} out of range for {} dimensions.".format(dim, ndim))
	return range(ndim)[dim]

def select(tensor, /, index, dim=0):
	args = [slice(None) for _ in range(tensor.ndim)]
	args[dim] = index
	return tensor[args]

def put(tensor, /, value, index, dim=0) -> None:
	args = [slice(None) for _ in range(tensor.ndim)]
	args[dim] = index
	tensor[args] = value

def moving_windows(tensor, /, size: int, dim: int):
	dim = range(tensor.ndim)[dim]
	leftpad = (size - size % 2) // 2
	rightpad = size - leftpad - 1
	pad_left = select(tensor, [0 for _ in range(leftpad)], dim=dim)
	pad_right = select(tensor, [-1 for _ in range(rightpad)], dim=dim)	
	tensor = torch.cat([pad_left, tensor, pad_right], dim=dim)
	return tensor.unfold(dim, size, 1)

def sma(tensor, /, size: int, dim=0):
	"""
		Simple moving average with replicate padding
	"""
	return moving_windows(tensor, size, dim).mean(dim=-1)

def gma(tensor, /, size: int, std: float, dim=0):
	"""
		Gaussian moving average with replicate padding
	"""
	x = 0.5 * (1 + torch.arange(-size, size, 2.0).to(tensor))
	kernel = 1 / (std * math.sqrt(math.tau)) * (-0.5 * (x / std)**2).exp()
	windows = moving_windows(tensor, kernel.numel(), dim)
	kernel = kernel * (kernel.numel() / kernel.sum())
	return (kernel * windows).mean(dim=-1)

def lerp_select(tensor, /, dim: int, indices, interpolation_fn=torch.lerp):
	indices = torch.as_tensor(indices)
	start = select(tensor, indices.floor().long(), dim=dim)
	stop = select(tensor, indices.ceil().long().clip(0, tensor.shape[dim]-1), dim=dim)
	weight = indices.fmod(1).view([-1 if s == tensor.shape[dim] else 1 for s in tensor.shape])
	return interpolation_fn(start, stop, weight)

def resample(tensor, /, nframes, dim=0, interpolation_fn=torch.lerp):
	if tensor.shape[dim] == nframes:
		return tensor
	else:
		indices = torch.arange(nframes).to(tensor) * (tensor.shape[dim] - 1) / (nframes - 1)
		return lerp_select(tensor, dim, indices, interpolation_fn=interpolation_fn)

def histmax(tensor, /, bins=None, xmin=float("-inf"), xmax=float("inf"), weight=None):
	if bins is None:
		bins = ((tensor >= xmin) & (tensor <= xmax)).sum().item() // 10
	
		if bins == 0:
			print(xmin, xmax)
			print(tensor.min(), tensor.max())
			# print((tensor >= xmin).sum(), (tensor <= xmax).sum(), tensor.shape)
			# print(((tensor >= xmin) & (tensor <= xmax)).sum(), tensor.shape)
	
	y, x = tensor.cpu().histogram(bins=bins, range=[xmin, xmax], weight=weight)	
	return x[y.argmax():][:2].mean().to(tensor.device)

def nonzero_ranges(tensor, /, dim=None):
	if tensor.ndim == 0:
		raise ValueError("Expected at least 1 dimension.")
	if dim is None:
		if tensor.ndim != 1:
			raise ValueError("Expected 1D tensor but got {}D.".format(tensor.ndim))
		dim = 0
	else:
		dim = parse_dim(dim, tensor.ndim)
	if tensor.ndim == 1:
		x = tensor != 0
		dx = x.diff(dim=0, prepend=~x[:1], append=~x[-1:])
		dx_idxs = dx.nonzero(as_tuple=False)[:, 0]
		return torch.as_tensor(list(zip(dx_idxs, dx_idxs[1:])))[~x[0]::2]		
	elif dim == 0:
		return [nonzero_ranges(tensor[:, i], 0) for i in range(tensor.shape[1])]
	else:
		return [nonzero_ranges(tensor[i], dim-1) for i in range(tensor.shape[0])]

## Dataset and loader helpers
class IterableDataset(torch.utils.data.Dataset):
	def __init__(self, iterable):
		super().__init__()
		self._items = tuple(iterable)

	def __len__(self):
		return self._items.__len__()

	def __iter__(self):
		return self._items.__iter__()
	
	def __getitem__(self, *args, **kwargs):
		return self._items.__getitem__(*args, **kwargs)

class BatchSampler:
	def __init__(self, length, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False, seed: int = None):
		self._batch_size = int(batch_size)
		self._nitems = length
		odd_last = self._nitems % self._batch_size != 0
		self._nbatches = math.ceil(self._nitems / self._batch_size) - (1 if odd_last and drop_last else 0)
		self._shuffle = bool(shuffle)
		self._drop_last = bool(drop_last)
		if seed is None:
			self._seed = torch.randint(torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max, [1])
		else:
			self._seed = torch.as_tensor([seed])

	def __len__(self):
		return self._nbatches

	def __iter__(self):
		if self._shuffle:
			with torch.random.fork_rng():
				torch.manual_seed(self._seed)
				self._seed += 1
				idxs = torch.randperm(self._nitems)
		else:
			idxs = torch.arange(self._nitems)
		batch_idxs = idxs.split(self._batch_size)
		if self._drop_last and batch_idxs[-1].shape[0] < self._batch_size:
			batch_idxs = batch_idxs[:-1]
		yield from batch_idxs

	def clone(self):
		return self.__class__(self._nitems, self._batch_size, self._shuffle, self._drop_last, self._seed)

class DictDatasetLoader:
	def __init__(self, dataset, *keys, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False, device=None):
		sampler = BatchSampler(len(dataset), batch_size, shuffle, drop_last)
		self._dataloaders = {key: torch.utils.data.DataLoader(dataset[key], batch_sampler=sampler.clone()) for key in keys}
		self._device = device

	def __len__(self):
		return len(self._dataloaders[list(self._dataloaders.keys())[0]])
	
	def _batch_mapping(self, batch):
		if self._device is None:
			return batch
		else:
			return batch.to(self._device)

	def _batches_to_dict(self, batches):
		return dict(zip(self._dataloaders.keys(), map(self._batch_mapping, batches)))
	
	def __iter__(self):
		batches = zip(*self._dataloaders.values())
		yield from map(self._batches_to_dict, batches)

class DictDataset(torch.utils.data.Dataset):
	def __init__(self, items):
		super().__init__()
		self._items = list(items)
		if any(not isinstance(item, dict) for item in self._items):
			index, item = next((index, item) for index, item in enumerate(self._items) if not isinstance(item, dict))
			raise ValueError("Items must be '{}' but got '{}' at position {}.".format(dict, type(item), index))
		self._keys = list(self._items[0].keys()) if self._items else tuple()
		if any(set(item.keys()) != set(self._keys) for item in self._items):
			index = next(index for index, item in enumerate(self._items) if set(item.keys()) != set(self._keys))
			raise ValueError("Items must be all have the same entries but got differences at position {}.".format(index))

	def __len__(self) -> int:
		return len(self._items)
	
	def __iter__(self):
		yield from self._items
		
	def keys(self) -> tuple:
		return tuple(self._keys)

	def __getitem__(self, arg):
		if arg in self.keys():
			return IterableDataset(item[arg] for item in self)
		elif isinstance(arg, int):
			return self._items[arg]
		elif hasattr(arg, "__iter__") and all(subarg in self.keys() for subarg in arg):
			if set(arg) == set(self.keys()):
				return self
			return DictDataset(tuple({key: item[key] for key in arg} for item in self))
		else:
			try:
				indices = torch.arange(len(self))[arg].view(-1).tolist()
			except:
				raise ValueError("Invalid argument '{}'.".format(arg))
			items = [self._items[index] for index in indices]
			if len(items) == len(self) and all(item1 is item2 for item1, item2 in zip(self._items, items)):
				return self
			else:
				return DictDataset(items)

	def __setitem__(self, arg, value):
		if isinstance(arg, str):
			items = list(value)
			if len(items) != len(self):
				raise ValueError("Expected '{}' items but got '{}'.".format(len(self), len(items)))
			for index in range(len(self)):
				self._items[index][arg] = items[index]
			if arg not in self._keys:
				self._keys.append(arg)
		elif isinstance(arg, int):
			if not isinstance(value, dict):
				raise ValueError("Expected '{}' but got '{}'.".format(dict, type(value)))
			if set(value.keys()) != set(self._keys):
				raise ValueError("Expected entries {} but got {}.".format(set(value.keys()), set(self._keys)))
			self._items[arg] = value
		elif hasattr(arg, "__iter__") and all(isinstance(subarg, str) for subarg in arg):
			args, items = list(arg), list(value)
			if len(items) != len(self):
				raise ValueError("Expected '{}' items but got '{}'.".format(len(self), len(items)))
			if any(len(item) != len(args) for item in items):
				index, item = next((index, item) for index, item in enumerate(items) if len(item) != len(args))
				raise ValueError("Expected '{}' entries at position {} but got '{}'.".format(len(args), index, len(item)))
			for index in range(len(self)):
				for key, value in zip(args, items[index]):
					self._items[index][key] = value
			for arg in args:
				if arg not in self._keys:
					self._keys.append(arg)
		else:
			items = list(value)
			if len(items) != len(self):
				raise ValueError("Expected '{}' items but got '{}'.".format(len(self), len(items)))
			if any(not isinstance(item, dict) for item in items):
				index, item = next((index, item) for index, item in items if not isinstance(item, dict))
				raise ValueError("Expected '{}' but got '{}' at position {}.".format(dict, type(item), index))
			if any(set(item.keys()) != set(self._keys) for item in items):
				index, item = next((index, item) for index, item in items if set(item.keys()) != set(self._keys))
				raise ValueError("Expected entries {} but got {} at position {}.".format(set(item.keys()), set(self._keys), index))
			try:
				indices = torch.arange(len(self))[arg].view(-1).tolist()
			except:
				raise ValueError("Invalid argument '{}'.".format(arg))
			if len(items) != len(indices):
				raise ValueError("Expected {} items but got {}.".format(len(indices), len(items)))
			for index, item in zip(indices, items):
				self._items[index] = item
	
	def shuffle(self, seed=None):
		if seed is None:
			return self[torch.randperm(len(self))]
		with torch.random.fork_rng():
			torch.manual_seed(seed)
			return self[torch.randperm(len(self))]

	def dataloader(self, **kwargs):
		return DictDatasetLoader(self, *self.keys(), **kwargs)

## Training helpers
class Schedule:
	def __init__(self, period, fn=lambda:None):
		self._period = int(period)
		self._fn = fn
	def occurs(self, timeline):
		return (timeline.iter + 1) % self._period == 0
	def __call__(self):
		self._fn()

class Timeline:
	def __init__(self, iterable, epochs: int, *schedules: tuple):
		self._iterable = iterable
		self._epochs = epochs
		self._schedules = list(schedules)
		self._iter_idx, self._item_idx, self._epoch_idx = 0, 0, 0

	@property
	def nitems(self):
		return len(self._iterable)
	@property
	def nepochs(self):
		return self._epochs
	@property
	def iter(self):
		return self._iter_idx
	@property
	def item(self):
		return self._item_idx
	@property
	def epoch(self):
		return self._epoch_idx
	
	def run(self):
		self._epoch_idx, self._iter_idx = 0, 0
		while self._epoch_idx < self._epochs:
			for item_idx, item in enumerate(self._iterable):
				self._item_idx = item_idx
				self.iteration(item)
				for schedule in self._schedules:
					if schedule.occurs(self):
						schedule()
				self._iter_idx += 1
			self._epoch_idx += 1
	
	def iteration(self, item):
		raise NotImplementedError()

## 3D angles helpers
class SO3:
	@classmethod
	def compose(cls, rmat1, rmat2):													# [...] x 3 x 3, [...] x 3 x 3
		return torch.matmul(rmat2, rmat1)											# [...] x 3 x 3
		return torch.matmul(y, x)
	
	@classmethod
	def inverse(cls, rmat):															# [...] x 3 x 3
		return rmat.transpose(-2, -1)												# [...] x 3 x 3

	@classmethod
	def to_local(cls, angles, topology):
		jidxs, pidxs = torch.as_tensor(topology.hierarchy(indices=True)).unbind(1)
		parents, joints = angles[..., pidxs, :, :], angles[..., jidxs, :, :]
		angles[..., jidxs, :, :] = cls.compose(joints, cls.inverse(parents))
		return angles

	@classmethod
	def to_global(cls, angles, topology):
		output = angles.clone()
		for j, p in topology.hierarchy(indices=True):
			parent, joint = output[..., p, :,:], output[..., j, :,:]
			output[..., j, :,:] = cls.compose(joint.clone(), parent.clone())
		return output

class SU2:
	@classmethod
	def identity(cls, /, *sizes, **kwargs):
		q = torch.zeros(*sizes, 4, **kwargs)										# [...] x 4
		q[..., 0] = 1.0																# [...]
		return q																	# [...] x 4

	@classmethod
	def to_SO3(cls, q, /):															# [...] x 4
		xyz2 = 2 * q[..., 1:]
		wx, xx = (q[..., :-2] * xyz2[..., [0]]).unbind(-1)
		wy, xy, yy = (q[..., :-1] * xyz2[..., [1]]).unbind(-1)
		wz, xz, yz, zz = (q * xyz2[..., [2]]).unbind(-1)
		return torch.stack([
			1.0 - (yy + zz), xy - wz, xz + wy,	
			xy + wz, 1.0 - (xx + zz), yz - wx,
			xz - wy, yz + wx, 1.0 - (xx + yy),
		], dim=-1).view(*q.shape[:-1], 3, 3)										# [...] x 3 x 3

	## Conversion methods
	@classmethod
	def from_SO3(cls, rmat, /):														# [...] x 3 x 3
		W_positive_half = rmat[..., [2,0,1], :][..., [1,2,0]].diagonal(0, -2, -1)	# [...] x 3
		W_negative_half = rmat[..., [1,2,0], :][..., [2,0,1]].diagonal(0, -2, -1)	# [...] x 3
		W = 0.5 * (W_positive_half - W_negative_half)								# [...] x 3	
		trace = rmat.diagonal(0, -2, -1).sum(dim=-1, keepdim=True)
		S = (trace + 1 + 1e-5).sqrt()												# [...] x 1
		q = torch.cat([0.5 * S, W / S], dim=-1)										# [...] x 4
		
		# Handle singularities
		flat_trace = trace.view(-1, 1)
		ntr = (flat_trace <= 0).nonzero(as_tuple=False)[:, 0]
		if ntr.numel() > 0:
			flat_q, flat_input = q.view(-1, 4), rmat.view(-1, 3, 3)				# N x 4, N x 3 x 3
			dmax, i = flat_input.diagonal(0, -2, -1).max(dim=-1, keepdim=False)
			i = i[ntr]
			S = 2 * (1 + 2 * dmax[ntr] - flat_trace[ntr].squeeze(-1)).sqrt()
			flat_q[ntr, 0], flat_q[ntr, i+1] = 2 * W.view(-1, 3)[ntr, i] / S, 0.25 * S
			flat_q[ntr, (i+1).fmod(3)+1] = (flat_input[ntr, i-2, i] + flat_input[ntr, i, i-2]) / S
			flat_q[ntr, (i+2).fmod(3)+1] = (flat_input[ntr, i, i-1] + flat_input[ntr, i-1, i]) / S
		
		# return normalized
		return cls.normalize(q)														# [...] x 4

	@classmethod
	def normalize(cls, q, /):														# [...] x 4
		return q / q.norm(p=2, dim=-1, keepdim=True)								# [...] x 4	

	@classmethod
	def slerp(cls, /, start, end, weight):
		"""
			Spherical Linear Interpolation as described in https://en.wikipedia.org/wiki/Slerp.
		"""
		# Parse inputs
		weight = torch.as_tensor(weight)
		weight = weight.unsqueeze(0) if weight.ndim == 0 else weight
		shape = torch.broadcast_shapes(start.shape[:-1], end.shape[:-1], weight.shape[:-1])
		start = start.broadcast_to(shape + start.shape[-1:])
		end = end.broadcast_to(shape + end.shape[-1:])
		weight = weight.broadcast_to(shape + weight.shape[-1:])
		
		# If the dot product is negative, slerp won't take the shorter path, fixed by reversing one quaternion
		dot = (start * end).sum(dim=-1, keepdim=True).clip(-1, 1) # [...] x 1
		start, dot = dot.sign() * start, dot.abs()
		
		# compute slerp
		angle = dot.acos()
		theta = weight * angle
		weight_y = torch.where(angle <= 1e-20, weight, theta.sin() / angle.sin())
		weight_x = theta.cos() - dot * weight_y
		return weight_x * start + weight_y * end

	@classmethod
	def to_local(cls, angles, topology):
		return cls.from_SO3(SO3.to_local(cls.to_SO3(angles), topology))

	@classmethod
	def to_global(cls, angles, topology):
		return cls.from_SO3(SO3.to_global(cls.to_SO3(angles), topology))

class so3:
	@classmethod
	def to_SO3(cls, input):																# [...] x 3
		angle = input.norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)						# [...] x 1 x 1
		W = torch.diag_embed(input, 0, -2, -1)[..., [1,2,0], :][..., :, [2,0,1]]		# [...] x 3 x 3
		W = W - W.transpose(-2, -1)														# [...] x 3 x 3
		W_coef = (angle.sin() / angle).masked_fill_(angle < 1e-20, 1.0)					# [...] x 1 x 1
		W2_coef = ((1 - angle.cos()) / angle**2).masked_fill_(angle < 3e-2, 0.5)		# [...] x 1 x 1
		output = W_coef * W + W2_coef * torch.matmul(W, W)								# [...] x 3 x 3
		output.diagonal(dim1=-2, dim2=-1)[:] += 1 # add identity matrix	
		return output

class HMat:
	@classmethod
	def identity(cls, *sizes, **kwargs):
		return torch.eye(4, **kwargs).broadcast_to(sizes + (4, 4)).clone()			# [...] x 4 x 4

	@classmethod
	def join(cls, rmat, tvec):														# [...] x 3 x 3, [...] x 3
		sizes = list(torch.broadcast_shapes(rmat.shape[:-2], tvec.shape[:-1])) 		# [...]
		tvec = tvec.unsqueeze(-1).broadcast_to(sizes + [3, 1])						# [...] x 3 x 1
		rmat = rmat.broadcast_to(sizes + [3, 3])									# [...] x 3 x 3
		hmat = torch.empty(sizes + [4, 4]).to(rmat)
		hmat[..., :3, :3] = rmat													# [...] x 3 x 3
		hmat[..., :3, 3:] = tvec													# [...] x 3 x 1
		hmat[..., 3:, :3] = 0.0														# [...] x 1 x 3
		hmat[..., 3:, 3:] = 1.0														# [...] x 1 x 1
		return hmat																	# [...] x 4 x 4	

	@classmethod
	def compose(cls, hmat1, hmat2):													# [...] x 4 x 4, [...] x 4 x 4
		return torch.matmul(hmat2, hmat1)											# [...] x 4 x 4
	
	@classmethod
	def rmat(cls, hmat):															# [...] x 4 x 4
		return hmat[..., :3, :3]													# [...] x 3 x 3

	@classmethod
	def tvec(cls, hmat):															# [...] x 4 x 4
		return hmat[..., :3, -1]													# [...] x 3

	@classmethod
	def split(cls, hmat):															# [...] x 4 x 4
		return cls.rmat(hmat), cls.tvec(hmat)										# [...] x 3 x 3, [...] x 3

	@classmethod
	def transform(cls, hmat, pts):													# [...] x 4 x 4, [...] x 3
		rmat, tvec = cls.split(hmat)												# [...] x 3 x 3, [...] x 3
		return torch.matmul(rmat, pts.unsqueeze(-1)).squeeze(-1) + tvec				# [...] x 3

