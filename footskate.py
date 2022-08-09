"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# UnderPressure
import anim, metrics, util
from data import FRAMERATE, TOPOLOGY, Contacts

# PyTorch
import torch

# extended topology with contact joints
EXTENDED_TOPOLOGY = anim.Topology(
	list(zip(TOPOLOGY.joints(), TOPOLOGY.parents())) + [
	("right_ankle_contacts",	["right_ankle"]),
	("right_foot_contacts",		["right_foot"]),
	("left_ankle_contacts",		["left_ankle"]),
	("left_foot_contacts",		["left_foot"]),
])

def get_velocity(trajectory):
	return trajectory.diff(dim=-3) * FRAMERATE # delta x / delta t

class Feet:
	JOINTS = [
		["left_foot_contacts", "right_foot_contacts"],
		["left_ankle_contacts", "right_ankle_contacts"],
	]
	JIDXS = torch.as_tensor([[EXTENDED_TOPOLOGY.index(joint) for joint in joints] for joints in JOINTS])	
	LUT = EXTENDED_TOPOLOGY.lut(TOPOLOGY)
	
	@classmethod
	def extend(cls, angles, skeleton, foot_heights):
		njoints = TOPOLOGY.njoints + 4
		
		# extend angles
		angles_local = torch.zeros(*angles.shape[:-2], njoints, 4).to(angles)
		util.put(angles_local, util.SU2.to_local(angles, TOPOLOGY), cls.LUT, dim=-2)
		util.put(angles_local, util.SU2.identity().to(angles_local), cls.JIDXS, dim=-2)
		angles = util.SU2.to_global(angles_local, EXTENDED_TOPOLOGY)
				
		# extend skeleton
		skeleton_local = torch.zeros(*skeleton.shape[:-2], njoints, skeleton.shape[-1]).to(skeleton)
		skeleton_local[..., cls.LUT, :] = anim.Positions.to_local(skeleton, TOPOLOGY)
		skeleton_local[..., cls.JIDXS, 2] = -foot_heights
		skeleton = anim.Positions.to_global(skeleton_local, EXTENDED_TOPOLOGY)
		
		return angles, skeleton

	@classmethod
	def reduce(cls, angles, skeleton):
		angles = util.select(angles, index=cls.LUT, dim=-2)
		skeleton = util.select(skeleton, index=cls.LUT, dim=-2)
		return angles, skeleton

class Cleaner:
	def __init__(self, model, iterations: int, qweight=1e-3, tweight=1e2, cweight=1e-5, fweight=5e-5, margin=0, device="cpu"):
		self._model = model
		self._niters = int(iterations)
		self._weights = dict(Q=float(qweight), T=float(tweight), C=float(cweight), F=float(fweight))
		self._margin = int(margin)
		self._device = device

	@property
	def qloss(self) -> bool:
		return isinstance(self._weights["Q"], float) and self._weights["Q"] > 0.0
	@property
	def tloss(self) -> bool:
		return isinstance(self._weights["T"], float) and self._weights["T"] > 0.0
	@property
	def closs(self) -> bool:
		return isinstance(self._weights["C"], float) and self._weights["C"] > 0.0
	@property
	def floss(self) -> bool:
		return isinstance(self._weights["F"], float) and self._weights["F"] > 0.0

	@property
	def device(self):
		return self._device

	@classmethod
	def smooth(cls, angles, skeleton, trajectory, size, std):
		targets = util.gma(anim.FK(angles, skeleton, trajectory, TOPOLOGY), size, std, dim=-3)
		angles = torch.nn.Parameter(angles)
		trajectory = torch.nn.Parameter(trajectory)
		optimiser = torch.optim.Adam([angles, trajectory], lr=1e-2)
		
		weights = dict(Q=1e-3, S=1e-1)
		for iter in range(50):
			# Compute loss
			losses = {}
			if weights["Q"] > 0.0:
				losses["Q"] = (angles.norm(p=2, dim=-1) - 1).square().mean()
			if weights["S"] > 0.0:
				positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
				losses["S"] = (positions - targets).norm(p=2, dim=-1).square().mean()
			loss = sum(losses[key] * value for key, value in losses.items())
			
			# optimise
			loss.backward()
			optimiser.step()
			optimiser.zero_grad()
		
		return util.SU2.normalize(angles.data), skeleton, trajectory.data

	@classmethod
	def sigmoid_like(cls, x, degree=2):
		m = (x > 0.5).float()
		s = 1 - 2 * m
		return m + 0.5 * s * (2 * (m + s * x))**degree

	def weights(self, t, m):
		w = self.sigmoid_like(torch.arange(m, device=self.device) / (m-1), degree=2)
		if t >= 2 * m:
			return torch.cat([w, torch.ones(t - 2 * m, device=self.device), (1 - w)])
		else:
			return torch.cat([w[:t//2+t%1], (1 - w)[-t//2:]])

	## Losses
	def quaternions_loss(self, angles):
		return (angles.norm(p=2, dim=-1) - 1).square().mean()		
	def trajectory_loss(self, trajectory, velocity_init):
		velocity = get_velocity(trajectory)
		return (velocity - velocity_init).square().sum(dim=-1).mean()
	def contacts_loss(self, positions, ranges, targets, weights):
		loss = 0.0
		for i in range(len(ranges)):
			for fb, lr in [[0, 0], [0, 1], [1, 0], [1, 1]]:
				for r, target in zip(ranges[i][fb][lr], targets[i][fb][lr]):
					dists2 = (positions[i, r[0]:r[1], Feet.JIDXS[fb, lr], :] - target).square().sum(dim=-1)
					loss += (weights[(r[1]-r[0]).item()] * dists2).mean()
		return loss	
	def vGRFs_loss(self, angles, skeleton, trajectory, vGRFs_init):
		positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
		vGRFs = self._model.vGRFs(positions)
		return metrics.MSLE(vGRFs, vGRFs_init)

	## Optimisation
	def __call__(self, angles, skeleton, trajectory, feet_heights=None):
		# Precomputations
		devices = dict(angles=angles.device, skeleton=skeleton.device, trajectory=trajectory.device)
		shape = angles.shape[:-3]
		angles = angles.reshape(-1, *angles.shape[-3:]).to(self.device)
		skeleton = skeleton.reshape(-1, *skeleton.shape[-3:]).to(self.device)
		trajectory = trajectory.reshape(-1, *trajectory.shape[-3:]).to(self.device)
		velocity_init = get_velocity(trajectory)									# N x F x 1 x 3
		positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
		
		# Regress feet heights if none
		if feet_heights is None:
			fh_coeffs = torch.as_tensor([0.096244677901268, 0.5001924633979797]).to(self.device)
			feet = skeleton[:, :, Contacts.JIDXS, :] # N x 1 x FB x LR x 3
			feet_lengths = feet.diff(dim=-3).squeeze(-3).norm(p=2, dim=-1).mean(dim=-1)
			feet_heights = (feet_lengths * fh_coeffs).unsqueeze(-1).expand(-1, -1, 2)
		else:
			feet_heights = feet_heights.reshape(-1, *feet_heights.shape[-3:]).to(self.device)

		
		# Predictions
		vGRFs_init = self._model.vGRFs(positions).detach()							# N x F x LR x 16
		contacts = Contacts.from_forces(vGRFs_init)									# N x F x FB x LR
		contact_ranges = util.nonzero_ranges(contacts, dim=-3)						# N x F x FB x LR

		# Prepare data
		angles, skeleton = Feet.extend(angles, skeleton, feet_heights)

		# Compute contact locations
		positions_init = anim.FK(angles, skeleton, trajectory, EXTENDED_TOPOLOGY)
		contact_weights = {}
		contact_locations = [[[[] for lr in [0, 1]] for fb in [0, 1]] for i in range(len(contact_ranges))]
		for i in range(len(contact_ranges)):
			for fb, lr in [[0, 0], [0, 1], [1, 0], [1, 1]]:
				for j in range(len(contact_ranges[i][fb][lr])):
					start, stop = contact_ranges[i][fb][lr][j].tolist()
					xy = positions_init[i, (start+stop)//2, Feet.JIDXS[fb, lr], :2]
					z = torch.zeros_like(xy[..., :1])
					contact_locations[i][fb][lr].append(torch.cat([xy, z], dim=-1))
					length = stop - start
					contact_weights[length] = contact_weights.get(length, self.weights(length, self._margin))
		
		angles = torch.nn.Parameter(angles)											# N x F x J x 4
		trajectory = torch.nn.Parameter(trajectory)									# N x F x 1 x 3
		optimiser = torch.optim.Adam([angles, trajectory], lr=1e-2)

		for iter in range(self._niters):
			# Compute loss
			losses = {}
			if self.qloss: ## Quaternion normalization
				losses["Q"] = self.quaternions_loss(angles)
			if self.closs or self.floss:
				angles_normalized = util.SU2.normalize(angles)
			if self.tloss: ## Trajectory velocity loss
				losses["T"] = self.trajectory_loss(trajectory, velocity_init)
			if self.closs: ## Contact position loss
				positions = anim.FK(angles_normalized, skeleton, trajectory, EXTENDED_TOPOLOGY)
				losses["C"] = self.contacts_loss(positions, contact_ranges, contact_locations, contact_weights)
			if self.floss: ## vGRFs invariance loss
				losses["F"] = self.vGRFs_loss(*Feet.reduce(angles_normalized, skeleton), trajectory, vGRFs_init)
			loss = sum(self._weights[key] * value for key, value in losses.items())
			
			# Optimise
			loss.backward()
			optimiser.step()
			optimiser.zero_grad()
		
		# remove contacts joints and smooth animation
		angles, skeleton = Feet.reduce(util.SU2.normalize(angles.data), skeleton)
		angles, skeleton, trajectory = self.smooth(angles, skeleton, trajectory.data, size=15, std=3)
		
		# Unflatten outputs
		trajectory = trajectory.reshape(*shape, *trajectory.shape[1:])
		angles = angles.reshape(*shape, *angles.shape[1:])
		skeleton = skeleton.reshape(*shape, *skeleton.shape[1:])
		return angles.to(devices["angles"]), skeleton.to(devices["skeleton"]), trajectory.to(devices["trajectory"])
