"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# UnderPressure
import anim, metrics, models, util
from data import FRAMERATE, TOPOLOGY, Dataset
from footskate import Cleaner
from skeletons import Skeletons
from visualization import MocapAndvGRFsApp

# PyTorch
import torch

def vGRFs_estimation(model, device, subject, sequence):
	testset = Dataset.testset("{}-{}".format(subject, sequence))
	for item in testset:
		angles, skeleton, trajectory, vGRFs_gt = item["angles"], item["skeleton"], item["trajectory"], item["forces"]
		positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
		vGRFs_pred = model.vGRFs(positions.to(device).unsqueeze(0)).squeeze(0).cpu()
		rmse = metrics.RMSE(vGRFs_pred.sum(dim=-1), vGRFs_gt.sum(dim=-1)).item()
		print("{:<55} RMSE = {:.1f}%".format("Subject {}, sequence '{}'".format(item["subject"], item["file"].stem), 100 * rmse))
		
		if len(testset) == 1:
			vGRFs_abs_error = (vGRFs_gt - vGRFs_pred).abs()
			MocapAndvGRFsApp(
				[(angles, skeleton, trajectory)],
				[vGRFs_gt, vGRFs_pred, vGRFs_abs_error],
				vGRF_labels=["Ground Truth", "Estimated", "Abs. Error"],
			).run()

def contacts_detection(model, device, subject, sequence):
	testset = Dataset.testset("{}-{}".format(subject, sequence))
	for item in testset:
		positions = anim.FK(item["angles"], item["skeleton"], item["trajectory"], TOPOLOGY)
		contacts_pred = model.contacts(positions.to(device).unsqueeze(0)).squeeze(0)	
		f1score = metrics.Fscore(contacts_pred, item["contacts"].to(device)).item()
		print("{:<55} F1 score = {:.3f}".format("Subject {}, sequence '{}'".format(item["subject"], item["file"].stem), f1score))

def retarget_to_underpressure(joint_positions, joint_names, niters, skeleton):
	# Define target joint positions
	joints = [joint for joint in TOPOLOGY if joint in joint_names]
	jidxs = list(map(TOPOLOGY.index, joints))
	target_jidxs = list(map(joint_names.index, joints))
	target = joint_positions[..., target_jidxs, :]
	shape, nframes = target.shape[:-3], target.shape[-3]
	
	# Prepare optimisation (target ~ scale * FK(angles, skeleton) + trajectory + translate)
	angles = torch.nn.Parameter(util.SU2.identity(*shape, nframes, len(TOPOLOGY)).to(target))
	trajectory = torch.nn.Parameter(joint_positions[..., [0], :].clone().to(target))
	translate = torch.nn.Parameter(torch.zeros(*shape, 1, 1, 3).to(target))
	scale = torch.nn.Parameter(torch.full([*shape, 1, 1, 1], 1.0).to(target))
	optimiser = torch.optim.Adam([angles, trajectory, translate, scale], lr=1e-1)
	skeleton = skeleton.to(target)
	p_weight = 1 / (skeleton[..., 2].amax(dim=-1) - skeleton[..., 2].amin(dim=-1)).mean().square()
	q_weight = 1e-3
	
	for iter in range(niters):
		# Compute global position from parameters
		positions = anim.FK(util.SU2.normalize(angles), skeleton, None, TOPOLOGY)[:, jidxs]
		positions = scale * positions + trajectory + translate

		# Compute losses
		p_error = (target - positions).norm(p=2, dim=-1).square().mean()
		q_error = (angles.norm(p=2, dim=-1) - 1).square().mean()
		loss = p_weight * p_error + q_weight * q_error
		
		# Optimize and log
		loss.backward()
		optimiser.step()
		optimiser.zero_grad()
		
	angles, trajectory, translate, scale = map(lambda x: x.data, [angles, trajectory, translate, scale])
	angles = util.SU2.normalize(angles)
	
	trajectory = (trajectory + translate) / scale
	
	return angles, trajectory

def contacts_detection_from_amass(model, joint_positions, framerate, skeleton):
	AMASS_JOINT_NAMES = [
		"pelvis",
		"left_hip",
		"right_hip",
		"spine_1",
		"left_knee",
		"right_knee",
		"spine_2",
		"left_ankle",
		"right_ankle",
		"neck",
		"left_foot",
		"right_foot",
		"head",
		"left_clavicle",
		"right_clavicle",
		"head_top",
		"left_shoulder",
		"right_shoulder",
		"left_elbow",
		"right_elbow",
		"left_wrist",
		"right_wrist",
		"left_finger_middle_3",
		"left_finger_thumb_3",
		"right_finger_middle_3",
		"right_finger_thumb_3",
	]
	
	# Retargeting to UnderPressure skeleton
	angles, trajectory = retarget_to_underpressure(
		joint_positions,
		AMASS_JOINT_NAMES,
		niters=150,
		skeleton=skeleton,
	)
	
	# resample angles and trajectory from input framerate 'framerate' to FRAMERATE
	out_nframes = round(trajectory.shape[-3] / framerate * FRAMERATE)
	angles = util.resample(angles, out_nframes, dim=-3, interpolation_fn=util.SU2.slerp)
	trajectory = util.resample(trajectory, out_nframes)
	
	# Predict contacts
	positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
	contacts = model.contacts(positions.unsqueeze(0)).squeeze(0).detach()
	contacts = util.resample(contacts.float(), joint_positions.shape[-3]) >= 0.5
	
	return contacts

def footskate_cleanup(model, device, index):
	item = torch.load("footskate_samples/{}.pt".format(index))
	cleaner = Cleaner(model, iterations=100, qweight=1e-3, tweight=1e2, cweight=1e-5, fweight=5e-5, margin=5, device=device)
	angles, skeleton, trajectory = cleaner(item["angles"], item["skeleton"], item["trajectory"])
	MocapAndvGRFsApp(
		[
			(item["angles"], item["skeleton"], item["trajectory"] - torch.as_tensor([0.5, 0.0, 0.0])),
			(angles, skeleton, trajectory + torch.as_tensor([0.5, 0.0, 0.0])),
		],
		motion_labels=["footskated", "cleaned"],
	).run()
	
if __name__ == "__main__":
	from argparse import ArgumentParser
	import sys
	parser = ArgumentParser()
	subparsers = parser.add_subparsers()
	
	parser_vGRFs = subparsers.add_parser("vGRFs")
	parser_vGRFs.add_argument("-subj", "-subject", default="*", help="Subject to be selected; default: *")
	parser_vGRFs.add_argument("-seq", "-sequence", default="*", help="Sequence to be selected; default: *")
	
	parser_contacts = subparsers.add_parser("contacts")
	parser_contacts.add_argument("-subj", "-subject", default="*", help="Subject to be selected; default: *")
	parser_contacts.add_argument("-seq", "-sequence", default="*", help="Sequence to be selected; default: *")
	
	parser_contacts_from_amass = subparsers.add_parser("contacts_from_amass")
	parser_contacts_from_amass.add_argument("-path", default="*", help="Path from which loading joint positions")
	parser_contacts_from_amass.add_argument("-framerate", type=float, help="Input framerate")
	parser_contacts_from_amass.add_argument("-skeleton", type=int, default=0, help="Index of the skeleton; default: 0")
	
	parser_cleanup = subparsers.add_parser("cleanup")
	parser_cleanup.add_argument("-idx", "-index", type=int, default=0, help="Index of the sequence to be selected; default: 0")
	
	parser.add_argument("-device", default="cuda")
	parser.add_argument("-checkpoint", default="pretrained.tar")
	
	args = parser.parse_args()
	
	# load model
	model = models.DeepNetwork(state_dict=torch.load(args.checkpoint)["model"]).to(args.device).eval()

	if sys.argv[1] == "vGRFs":
		vGRFs_estimation(model, args.device, args.subj, args.seq)
	elif sys.argv[1] == "contacts":
		contacts_detection(model, args.device, args.subj, args.seq)
	elif sys.argv[1] == "contacts_from_amass":
		joint_positions = torch.load(args.path).to(args.device)
		framerate = args.framerate
		skeleton = Skeletons.all()[args.skeleton].to(args.device)
		contacts = contacts_detection_from_amass(model, joint_positions, framerate, skeleton)
	elif sys.argv[1] == "cleanup":
		footskate_cleanup(model, args.device, args.idx)






