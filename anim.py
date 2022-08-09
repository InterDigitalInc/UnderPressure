"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# Project
import util

# Misc
import torch

class Topology:
	def __init__(self, hierarchy: list):
		# joints
		self._joints = tuple(joint for joint, _ in hierarchy)
		self._joints_idxs = {joint:index for index, joint in enumerate(self._joints)}
		
		# parents
		self._parents = tuple(tuple(parents) for _, parents in hierarchy)
		self._parents_idxs = tuple(tuple(self.index(parent) for parent in parents) for parents in self._parents)
		
		# children
		self._children = tuple(set() for joint in self._joints)
		for joint, parents_idxs in zip(self._joints, self._parents_idxs):
			for pidx in parents_idxs:
				self._children[pidx].add(joint)
		self._children = tuple(frozenset(children) for children in self._children)
		self._children_idxs = tuple(frozenset(self.index(child) for child in children) for children in self._children)
		
		# roots
		self._roots = tuple(self[jidx] for jidx, parents in enumerate(self._parents) if len(parents) == 0)
		self._roots_idxs = tuple(self.index(root) for root in self._roots)
		
		# depth
		self._depths = [frozenset() for _ in self._joints]
		pending = [(ridx, 0) for ridx in self._roots_idxs]
		while len(pending) > 0:
			jidx, depth = pending.pop(0)
			self._depths[jidx] |= {depth}
			pending += [(cidx, depth + 1) for cidx in self._children_idxs[jidx]]
		self._depths = tuple(self._depths)
		
		# hierarchy, i.e. list of (joint, first parent) pairs in descending order
		self._hierarchy = []
		for joint, parents in zip(self._joints, self._parents):
			if len(parents) != 0:
				self._hierarchy.append((joint, parents[0]))
		self._hierarchy = sorted(self._hierarchy, key=lambda b: max(self._depths[self.index(b[1])]))
		self._hierarchy_idxs = [(self.index(j), self.index(p)) for j, p in self._hierarchy]

		# bones
		self._bones = []
		for joint, parents in zip(self._joints, self._parents):
			self._bones += [(joint, parent) for parent in parents]
		self._bones_idxs = [(self.index(joint), self.index(parent)) for joint, parent in self._bones]
		
	def __len__(self) -> int:
		return self.njoints

	@property
	def njoints(self) -> int:
		return len(self._joints)

	@property
	def nbones(self) -> int:
		return len(self._bones)
		
	def __contains__(self, joint: str) -> bool:
		return joint in self._joints

	def __getitem__(self, index: int) -> str:
		return self._joints[index]

	def isroot(self, joint: str) -> bool:
		return self.parent(joint, index=False) is None

	def isequiv(self, other) -> bool:
		assert isinstance(other, Topology), "Expected {} but got {}.".format(Topology, type(other))
		return self.renamed(map(str, range(len(self)))) == other.renamed(map(str, range(len(other))))
		
	def issubset(self, other) -> bool:
		assert isinstance(other, Topology), "Expected {} but got {}.".format(Topology, type(other))
		return all(joint in other for joint in self)

	def index(self, joint: str) -> int:
		return self._joints_idxs[joint]

	def joints(self):
		return self._joints
	
	def parent(self, joint, index=None):
		if index is None:
			index = isinstance(joint, int)
		if not isinstance(joint, int):
			joint = self.index(joint)
		parents = self._parents_idxs[joint] if index else self._parents[joint]
		return (-1 if index else None) if len(parents) == 0 else parents[0]
	
	def parents(self, indices=False):
		return self._parents_idxs if indices else self._parents

	def roots(self, indices=False):
		return self._roots_idxs if indices else self._roots

	def hierarchy(self, indices=False):
		return self._hierarchy_idxs.copy() if indices else self._hierarchy.copy()

	def bones(self, indices=False):
		return self._bones_idxs.copy() if indices else self._bones.copy()

	def lutable(self, other) -> list:
		if not isinstance(other, Topology):
			raise TypeError("Expected type '{}' but got '{}'.".format(Topology, type(other)))
		return set(other._joints).issubset(self._joints)

	def lut(self, other) -> list:
		if not self.lutable(other):
			raise ValueError("Target topology must be a subset.")
		return [self.index(joint) for joint in other._joints]
		
	def mirrored(self):
		def mirrored(joint: str):
			return joint.replace("left", "<placeholder>").replace("right", "left").replace("<placeholder>", "right")
		return Topology([(mirrored(j), [mirrored(p) for p in parents]) for j, parents in zip(self._joints, self._parents)])

class Positions:
	def to_local(positions, topology: Topology):
		output = positions.clone()													# [...] x J x 3
		jidxs, pidxs = torch.as_tensor(topology.hierarchy(indices=True)).unbind(1)
		output[..., jidxs, :] -= output[..., pidxs, :]
		return output																# [...] x J x 3
		
	def to_global(positions, topology: Topology):
		output = positions.clone()													# [...] x J x 3
		for j, p in topology.hierarchy(indices=True):
			output[..., j, :] += output[..., p, :]									# [...]	  x 3
		return output																# [...] x J x 3

def FK(angles: torch.Tensor, skeleton: torch.Tensor, trajectory: torch.Tensor, topology: Topology) -> torch.Tensor:
	"""
		Compute global joint positions (i.e. forward kinematics) from MVNX data representation.
	"""
	output = util.HMat.join(
		util.SO3.to_local(util.SU2.to_SO3(angles), topology),
		Positions.to_local(skeleton, topology),
	)
	for j, p in topology.hierarchy(indices=True):
		parent, joint = output[..., p, :,:], output[..., j, :,:]
		output[..., j, :,:] = util.HMat.compose(joint.clone(), parent.clone())
	return util.HMat.tvec(output) + trajectory
