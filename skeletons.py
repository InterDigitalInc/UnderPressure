"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# UnderPressure
from data import TOPOLOGY, Subject
from anim import Positions

# PyTorch
import torch

class Skeletons:
	@classmethod
	def all(cls):
		files = [next(s.preprocessed_files()) for s in Subject.all()]
		return torch.cat([torch.load(file)["skeleton"] for file in files])

	@classmethod
	def train(cls):
		files = [next(s.preprocessed_files()) for s in Subject.train()]
		return torch.cat([torch.load(file)["skeleton"] for file in files])

	@classmethod
	def test(cls):
		files = [next(s.preprocessed_files()) for s in Subject.test()]
		return torch.cat([torch.load(file)["skeleton"] for file in files])

class SvdSkeletonSpace:
	def __init__(self, skeletons, asymmetric=False):
		# representation
		ridxs = TOPOLOGY.roots(indices=True)
		self._jidxs = [i for i in range(len(TOPOLOGY)) if i not in ridxs]
		
		# handle LR symmetry
		self._asymmetric = bool(asymmetric)
		sorted_joints = sorted(enumerate(TOPOLOGY), key=lambda t: t[1])
		self._left_jidxs = [i for i, j in sorted_joints if "left" in j]
		self._right_jidxs = [i for i, j in sorted_joints if "right" in j]
		
		# Compute bone lengths and unit offsets
		offsets = Positions.to_local(skeletons, TOPOLOGY)[..., self._jidxs, :]		# S x J x 3
		lengths = offsets.norm(p=2, dim=-1, keepdim=True)							# S x J x 1
		self._unit_offsets = torch.zeros(len(TOPOLOGY), 3)							# J x 3
		self._unit_offsets[self._jidxs, :] = offsets.where(lengths <= 1e-10, offsets / lengths).mean(dim=0)
		
		# Svd Basis
		svd_samples = lengths.squeeze(-1)
		self._mean = svd_samples.mean(dim=0)										# C
		svd_samples -= self._mean
		V = torch.linalg.svd(svd_samples)[2]										# C x C
		basis = svd_samples * V.unsqueeze(1)										# C x N x C
		self._basis = basis.sum(dim=2).std(dim=1, keepdim=True) * V					# C x C
		
	@property
	def asymmetric(self) -> bool:
		return self._asymmetric

	def from_params(self, params):													# [...] x C
		mean, basis = self._mean.to(params), self._basis.to(params)
		lengths = mean + (params.unsqueeze(-1) * basis).sum(dim=-2)					# [...] x C
		scales = torch.ones_like(params[..., len(TOPOLOGY)*[0]])			# [...] x J
		scales[..., self._jidxs] = lengths
		if not self.asymmetric:
			scales[..., self._right_jidxs] = scales[..., self._left_jidxs]
		offsets = scales.unsqueeze(-1) * self._unit_offsets.to(scales)				# [...] x J x 3
		return Positions.to_global(offsets, TOPOLOGY)

	def sample(self, *sizes, mean=0.0, std=1.0, **kwargs):
		normal_sample = torch.randn(*sizes, self._basis.shape[0], **kwargs)
		mean = torch.as_tensor(mean).to(normal_sample)
		std = torch.as_tensor(std).to(normal_sample)
		params = mean + std * normal_sample		
		return self.from_params(params)

	def sample_like(self, input, *args, **kwargs):
		kwargs = dict(dtype=input.dtype, layout=input.layout, device=input.device) | kwargs
		return self.sample(*input.shape[:-2], *args, **kwargs)

class SkeletonSampler:
	def __init__(self, offsets_std, lengths_std, asymmetric, skeletons):
		self._offsets_std, self._lengths_std = float(offsets_std), float(lengths_std)
		self._asymmetric = bool(asymmetric)
		
		# Build underlying skeleton sampler
		self._sampler = SvdSkeletonSpace(Positions.to_local(skeletons, TOPOLOGY))
		
		# Computer joints indexing helpers
		sorted_joints = sorted(enumerate(TOPOLOGY), key=lambda t: t[1])
		self._center_jidxs = [i for i, j in sorted_joints if not ("left" in j or "right" in j)]
		self._left_jidxs = [i for i, j in sorted_joints if "left" in j]
		self._right_jidxs = [i for i, j in sorted_joints if "right" in j]

		# Precompute priviledged unit LR axis
		offsets = self._sampler.sample()
		left_offsets = offsets[..., self._left_jidxs, :]
		right_offsets = offsets[..., self._right_jidxs, :]
		l2r_offsets = right_offsets - left_offsets
		l2r_offsets_norms = l2r_offsets.norm(p=2, dim=-1, keepdim=True)
		mask = l2r_offsets_norms > 1e-3 * l2r_offsets_norms.amax(dim=-2, keepdim=True)
		l2r_offsets = l2r_offsets[mask.expand_as(l2r_offsets)].view(*l2r_offsets.shape[:-2], -1, l2r_offsets.shape[-1])
		l2r_unit = l2r_offsets.sum(dim=-2, keepdim=True)
		self._l2r_unit = (1e3 * l2r_unit / l2r_unit.norm(p=2, dim=-1, keepdim=True)).round() / 1e3

	@classmethod
	def train(cls, offsets_std: float, lengths_std: float, asymmetric=False):
		return cls(offsets_std, lengths_std, asymmetric, Skeletons.train())
	@classmethod
	def test(cls, offsets_std: float, lengths_std: float, asymmetric=False):
		return cls(offsets_std, lengths_std, asymmetric, Skeletons.test())

	@property
	def asymmetric(self) -> bool:
		return self._asymmetric
	
	def _lengths(self, offsets, std):
		normal_samples = torch.randn(*offsets.shape[:-1], 1).to(offsets)			# [...] x J x 1
		scales = 1 + std * normal_samples.clamp(-4.0, 4.0)							# [...] x J x 1
		if not self.asymmetric:
			scales[..., self._right_jidxs, :] = scales[..., self._left_jidxs, :]
		return scales * offsets

	def _offsets(self, offsets, std):
		# sample delta offsets
		lengths = offsets.norm(p=2, dim=-1, keepdim=True)							# [...] x J x 1
		normal_samples = torch.randn_like(offsets)									# [...] x J x 3
		doffsets = std * lengths * normal_samples.clamp(-4.0, 4.0)					# [...] x J x 1
		
		if not self.asymmetric:
			def project(a, b):
				return ((a * b).sum(dim=-1, keepdim=True) / b.square().sum(dim=-1, keepdim=True)) * b
			
			l2r_unit = self._l2r_unit.to(offsets)
			
			# Project centered joint delta offsets onto LR axis
			center_doffsets = doffsets[..., self._center_jidxs, :]
			doffsets[..., self._center_jidxs, :] = center_doffsets - project(center_doffsets, l2r_unit)
			
			# Make left/right joint delta offsets symmetric w.r.t. LR axis
			left_doffsets = doffsets[..., self._left_jidxs, :]
			doffsets[..., self._right_jidxs, :] = left_doffsets - 2 * project(left_doffsets, l2r_unit)
		
		# keep same length
		offsets = offsets + doffsets
		scale = (lengths / offsets.norm(p=2, dim=-1, keepdim=True)).nan_to_num(0.0)
		return scale * offsets

	def sample(self, *args, **kwargs):
		offsets = self._sampler.sample(*args, **kwargs)
		if self._offsets_std > 0.0:
			offsets = self._offsets(offsets, self._offsets_std)
		if self._lengths_std > 0.0:
			offsets = self._lengths(offsets, self._lengths_std)
		return self._sampler.repr.to(self.repr, offsets)

	def sample_like(self, *args, **kwargs):
		offsets = self._sampler.sample_like(*args, **kwargs)
		if self._offsets_std > 0.0:
			offsets = self._offsets(offsets, self._offsets_std)
		if self._lengths_std > 0.0:
			offsets = self._lengths(offsets, self._lengths_std)
		return Positions.to_global(offsets, TOPOLOGY)
