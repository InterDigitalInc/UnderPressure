"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# UnderPressure
import anim, metrics, models, util
from data import TOPOLOGY, Contacts, Dataset
from skeletons import SkeletonSampler

# Python
import math, time
from pathlib import Path

# PyTorch
import torch

MIRROR_LUT = TOPOLOGY.lut(TOPOLOGY.mirrored())

def split(dataset, ratio, window_length):
	# select random non-overlapping windows for validation set
	nframes = sum(item["angles"].shape[-3] for item in dataset)
	valid_nwindows =  int((1 - ratio) * nframes) // window_length + 1
	windows = []
	for index, item in enumerate(dataset):
		starts = torch.arange(0, item["angles"].shape[-3] - 2 * window_length + 1, window_length)
		indices = torch.full_like(starts, index)
		windows.append(torch.stack([indices, starts], dim=-1))
	windows = torch.cat(windows)
	windows = windows[torch.randperm(windows.shape[0])[:valid_nwindows]]
	
	# split according to selected windows
	valid_items, train_items = [], []
	for index, item in enumerate(dataset):
		valid_starts = windows[windows[:, 0] == index, 1].sort()[0]
		starts = torch.cat([valid_starts, torch.as_tensor([0]), valid_starts + window_length])
		stops = torch.cat([valid_starts + window_length, valid_starts, torch.as_tensor([item["angles"].shape[-3]])])
		items = dataset.slices(index, starts, stops)
		train_items += [item for item in items[len(valid_starts):] if item["angles"].shape[-3] > 0]
		valid_items += items[:len(valid_starts)]
	return Dataset(train_items), Dataset(valid_items)

def prepare(split_ratio, sequence_length, sequence_overlap):
	# split
	dataset = Dataset.trainset()["angles", "skeleton", "trajectory", "contacts", "forces"]
	trainset, validset = split(dataset, split_ratio, sequence_length)

	# slice into overlapping windows
	if set(a.shape[-3] for a in trainset["angles"]) != {sequence_length}:
		trainset = trainset.windowed(sequence_length, sequence_overlap)
	if set(a.shape[-3] for a in validset["angles"]) != {sequence_length}:
		validset = validset.windowed(sequence_length, 0)

	return trainset.shuffle(), validset

def rnd_transform(positions, forces):												# N x F x J x 3, N x F x 2 x 16
	bs, device = positions.shape[0], positions.device
	transform = util.HMat.identity(bs, 1, 1, device=device) 						# N x 1 x 1 x 4 x 4

	# Normalize positions
	translate = -positions.mean(dim=(1,2), keepdim=True)							# N x 1 x 1 x 3
	translate = util.HMat.join(torch.eye(3, device=device), translate)				# N x 1 x 1 x 4 x 4
	transform = util.HMat.compose(transform, translate)								# N x 1 x 1 x 4 x 4
	
	# XY-plane rotations
	angles = math.tau * torch.rand(bs, 1, 1, 1, device=device)						# N x 1 x 1 x 1
	rotation = util.so3.to_SO3(angles * torch.as_tensor([0, 0, 1]).to(angles))		# N x 1 x 1 x 3 x 3
	rotation = util.HMat.join(rotation, torch.zeros(3, device=device))				# N x 1 x 1 x 4 x 4
	transform = util.HMat.compose(transform, rotation)								# N x 1 x 1 x 4 x 4
	
	# Scale
	scales = (1 + 0.25 * torch.randn(bs, 1, 1, 1, device=device)).clamp(min=0.1)	# N x 1 x 1 x 1
	scales = scales.expand(*scales.shape[:-1], 3)									# N x 1 x 1 x 3
	scaling = torch.diag_embed(scales, dim1=-2, dim2=-1)							# N x 1 x 1 x 3 x 3
	scaling = util.HMat.join(scaling, torch.zeros(3, device=device))				# N x 1 x 1 x 4 x 4
	transform = util.HMat.compose(transform, scaling)								# N x 1 x 1 x 4 x 4
	
	# Translate
	translate = 10.0 * torch.randn(bs, 1, 1, 3, device=device)						# N x 1 x 1 x 3
	translate = util.HMat.join(torch.eye(3, device=device), translate)				# N x 1 x 1 x 4 x 4
	transform = util.HMat.compose(transform, translate)								# N x 1 x 1 x 4 x 4
	
	# Apply transform on positions
	positions = util.HMat.transform(transform, positions)
	
	# Mirrorring
	mirror = torch.rand(bs, device=device) < 0.5									# N
	positions_mirrored = positions.clone()
	positions_mirrored[mirror] = positions[mirror][..., MIRROR_LUT, :]
	forces_mirrored = forces.clone()
	forces_mirrored[mirror] = forces[mirror][..., [1, 0], :]
	
	return positions_mirrored, forces_mirrored										# N x F x J x 3, N x F x 2 x 16

class Trainer(util.Timeline):	
	def __init__(self, **kwargs):
		self.device = kwargs["device"]
		
		# model and optimiser
		self.model = models.DeepNetwork()
		self.model = self.model.initialize().to(self.device)
		self.optimiser = torch.optim.Adam(self.model.parameters(), lr=kwargs["learning_rate"])
		self.msle_weight = kwargs["msle_weight"]
		
		# data preparation
		trainset, validset = prepare(kwargs["split_ratio"], kwargs["sequence_length"], kwargs["sequence_overlap"])
		
		self._skeletons_std = kwargs["skeletons_basis_std"]
		self._skeletons_sampler = SkeletonSampler.train(kwargs["skeletons_offsets_std"], kwargs["skeletons_lengths_std"])
		trainset = trainset["angles", "skeleton", "trajectory", "forces"]
	
		dataloader = trainset.dataloader(
			batch_size=kwargs["batch_size"],
			shuffle=True,
			device=self.device,
		)
		
		# prepare validset: set motions in model representation
		angles = torch.stack(list(validset["angles"]))
		skeletons = torch.stack(list(validset["skeleton"]))
		trajectory = torch.stack(list(validset["trajectory"]))
		self.validset = dict(
			positions=	anim.FK(angles, skeletons, trajectory, TOPOLOGY),
			contacts=	torch.stack(list(validset["contacts"])),
			forces=		torch.stack(list(validset["forces"])),
		)
		
		# logging support
		self.ckp = kwargs["ckp"]
		
		# instanciate timeline
		num_epochs = int(kwargs["iterations"] / len(dataloader) + 0.5)
		super().__init__(dataloader, num_epochs, *[
			util.Schedule(period=100,	fn=self._losses_logging),	# log loss values every X batches
			util.Schedule(period=3000,	fn=self._validation),		# validation every X batches
		])
		
	def iteration(self, batch):
		angles, skeleton, trajectory, forces = batch["angles"], batch["skeleton"], batch["trajectory"], batch["forces"]
		if self._skeletons_std != 0.0:
			skeleton = self._skeletons_sampler.sample_like(skeleton, std=self._skeletons_std)
			
		# compute global joint positions
		positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
	
		# apply random vGRF-invariant transformations
		positions, forces_target = rnd_transform(positions, forces)
		
		# make predictions and compute loss
		forces_pred = self.model.vGRFs(positions)
		self.msle = metrics.MSLE(forces_pred, forces_target)
		loss = self.msle_weight * self.msle
	
		# optimize
		loss.backward()
		self.optimiser.step()
		self.optimiser.zero_grad()

	def _losses_logging(self):
		item, epoch = self.item + 1, self.epoch + 1
		print("[{}/{}][{}/{}]   MSLE = {:.5e}".format(item, self.nitems, epoch, self.nepochs, self.msle))
	
	def _validation(self):
		print("Validation #{}".format(self.iter))
		
		# Make predictions
		with self.model.frozen():
			forces_pred = []
			for positions in self.validset["positions"].split(128):
				forces_pred.append(self.model.vGRFs(positions.to(self.device)).detach().cpu())
			forces_pred = torch.cat(forces_pred)
			
		# compute and log msle and F1 score
		contacts_rec = torch.cat([Contacts.from_forces(f) for f in forces_pred.split(128)])
		msle = metrics.MSLE(forces_pred, target=self.validset["forces"]).item()
		f1score = metrics.Fscore(contacts_rec, self.validset["contacts"])
		if not hasattr(self, "best_f1score") or f1score >= self.best_f1score:
			self.best_f1score = f1score
			torch.save(dict(model=self.model.state_dict()), self.ckp)
		print("MSLE={:.3e} F1={:.5f}".format(msle, f1score))

if __name__ == "__main__":
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("-ckp", default="./checkpoint.tar", type=Path,			help="Path to make checkpoint during training ........................ default: 'checkpoint.tar'")
	parser.add_argument("-device", default="cuda", type=str,					help="Device used to run training .................................... default: cuda")
	parser.add_argument("-learning_rate", default=3e-5, type=float, 			help="Adam optimisation algorithm learning rate ...................... default: 3e-5")
	parser.add_argument("-skeletons_basis_std", default=2.0, type=float,		help="Data augmentation skeletons SVD basis standard deviation ....... default: 2.0")
	parser.add_argument("-skeletons_offsets_std", default=0.0075, type=float,	help="Data augmentation skeleton joint offsets standard deviation .... default: 0.0075")
	parser.add_argument("-skeletons_lengths_std", default=0.0150, type=float,	help="Data augmentation skeleton bone lengths standard deviation ..... default: 0.0150")
	parser.add_argument("-msle_weight", default=0.002, type=float,				help="MSLE loss weight ............................................... default: 0.002")
	parser.add_argument("-batch_size", default=64, type=int,					help="Batch size ..................................................... default: 64")
	parser.add_argument("-iterations", default=1e8, type=int,					help="Number of training iterations .................................. default: 1e8")
	parser.add_argument("-split_ratio", default=0.9, type=float,				help="Train/Validation split ratio ................................... default: 0.9")
	parser.add_argument("-sequence_length", default=240, type=int,				help="Training sequences length ...................................... default: 240")
	parser.add_argument("-sequence_overlap", default=239, type=int,					help="Training sequences overlap ................................. default: 239")
	Trainer(**vars(parser.parse_args())).run()
