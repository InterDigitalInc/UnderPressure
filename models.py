"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# UnderPressure
from data import TOPOLOGY, Contacts

# PyTorch
import torch

class HeuristicThresholds(torch.nn.Module):
	def __init__(self, height: float = 0.0, velocity: float = 0.0):
		super().__init__()
		self.register_buffer("_height", torch.full([1], height), persistent=True)
		self.register_buffer("_velocity", torch.full([1], velocity), persistent=True)
		
	@property
	def height(self):
		return self._height
	@property
	def velocity(self):
		return self._velocity

	def contacts(self, positions, ground=None):
		feet = positions[..., Contacts.JIDXS, :]
		heights = feet[..., 2]
		if ground is not None:
			heights = heights - ground
		velocities = feet.diff(dim=-4, prepend=feet[..., [1], :, :, :]).norm(p=2, dim=-1)
		return (heights < self.height).logical_and_(velocities < self.velocity)	

	@classmethod
	def optimal(cls):
		return cls(height=0.11423380672931671, velocity=0.009324918501079082)

class Transpose(torch.nn.Module):
	def __init__(self, dim1, dim2):
		super().__init__()
		self._dim1, self._dim2 = dim1, dim2

	def extra_repr(self):
		return "{}, {}".format(self._dim1, self._dim2)

	def forward(self, input):
		return input.transpose(self._dim1, self._dim2)

class DeepNetwork(torch.nn.Sequential):
	def __init__(self, cnn_kernel=7, cnn_dropout=0.0, fc_depth=3, fc_dropout=0.2, state_dict=None):
		super().__init__()
		cnn_features = [3 * TOPOLOGY.njoints, 128, 128, 256, 256]
		features_out = 16
		
		## Preprocess part
		pre_layers = [																# N x F x J x [...]
			torch.nn.Flatten(start_dim=2, end_dim=-1),								# N x F x C
			Transpose(-2, -1),														# N x C x F
		]		
		
		## Convolutional part
		conv = lambda c_in, c_out: torch.nn.Conv1d(c_in, c_out, cnn_kernel, padding=cnn_kernel//2, padding_mode="replicate")
		cnn_layers = []
		for c_in, c_out in zip(cnn_features[:-1], cnn_features[1:]):				# N x C x F
			cnn_layers += [
				torch.nn.Dropout(p=cnn_dropout),									# N x Ci x F
				conv(c_in, c_out),													# N x Ci x F
				torch.nn.ELU(),														# N x Ci x F
			]
		
		## Fully connected part
		fc_layers = [Transpose(-2, -1)]												# N x F x Cn
		for _ in range(fc_depth - 1):
			fc_layers += [															# N x F x Ci
				torch.nn.Dropout(p=fc_dropout),										# N x F x Ci
				torch.nn.Linear(cnn_features[-1], cnn_features[-1]),				# N x F x Ci
				torch.nn.ELU()														# N x F x Ci
			]
		fc_layers += [																# N x F x Ci
			torch.nn.Dropout(p=fc_dropout),											# N x F x 2*Co
			torch.nn.Linear(cnn_features[-1], 2*features_out, bias=False),			# N x F x 2*Co
			torch.nn.Unflatten(-1, (2, features_out)),								# N x F x 2 x Co
			torch.nn.Softplus(),													# N x F x 2 x Co
		]
		super().__init__(*pre_layers, *cnn_layers, *fc_layers)
		
		if state_dict:
			self.load_state_dict(state_dict)

	def initialize(self):
		GAINS = {
			torch.nn.Sigmoid:	torch.nn.init.calculate_gain("sigmoid"),
			torch.nn.ReLU:		torch.nn.init.calculate_gain("relu"),
			torch.nn.LeakyReLU:	torch.nn.init.calculate_gain("leaky_relu"),
			torch.nn.ELU:		torch.nn.init.calculate_gain("relu"),
			torch.nn.Softplus:	torch.nn.init.calculate_gain("relu"),
		}
		for layer, activation in zip(list(self)[:-1], list(self)[1:]):
			if len(list(layer.parameters())) > 0 and type(activation) in GAINS:
				if not isinstance(activation, type):
					activation = type(activation)
				if activation not in GAINS:
					raise Exception("Initialization not defined for activation '{}'.".format(type(activation)))
				if isinstance(layer, torch.nn.Linear):
					torch.nn.init.xavier_normal_(layer.weight, GAINS[activation])
					if layer.bias is not None:
						torch.nn.init.zeros_(layer.bias)
				elif isinstance(layer, torch.nn.Conv1d):
					torch.nn.init.xavier_normal_(layer.weight, GAINS[activation])
					if layer.bias is not None:
						torch.nn.init.zeros_(layer.bias)
				else:
					raise Exception("Initialization not defined for layer '{}'.".format(type(layer)))
		return self

	def vGRFs(self, positions):
		return self(positions)
		
	def contacts(self, positions):
		return Contacts.from_forces(self.vGRFs(positions))

