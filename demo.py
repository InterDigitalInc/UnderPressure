"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# UnderPressure
import anim, metrics, models
from data import TOPOLOGY, Dataset
from footskate import Cleaner
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
	elif sys.argv[1] == "cleanup":
		footskate_cleanup(model, args.device, args.idx)






