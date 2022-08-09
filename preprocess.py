"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# UnderPressure
import parsing, util
from data import Subject, Forces, Contacts

# Python
import logging
_logger = logging.getLogger(__name__)

# PyTorch
import torch

def parse_sequence(subject, sequence, dst):
	# parse data
	mocap = parsing.MVNX.parse(subject.mocap_mvnx_file(sequence))
	insoles = parsing.MoticonTXT.parse(subject.insoles_txt_file(sequence))
	sync_pts = parsing.SyncCSV.parse(subject.sync_csv_file(sequence))

	# synchronizsation & resampling
	nframes = round((sync_pts["mocap"][1] - sync_pts["mocap"][0]) * insoles.pop("framerate") / mocap.pop("framerate"))
	def indices(start, stop):
		return torch.arange(nframes) / nframes * (stop - start) + start
	mocap["angles"] = util.lerp_select(mocap["angles"], dim=0, indices=indices(*sync_pts["mocap"]), interpolation_fn=util.SU2.slerp)
	mocap["skeleton"] = mocap["skeleton"].unsqueeze(0)
	mocap["trajectory"] = util.lerp_select(mocap["trajectory"], dim=0, indices=indices(*sync_pts["mocap"]))
	for key in insoles:
		if isinstance(insoles[key], torch.Tensor):
			insoles[key] = torch.stack([
				util.lerp_select(insoles[key][:, 0], dim=0, indices=indices(*sync_pts["left_insole"])),
				util.lerp_select(insoles[key][:, 1], dim=0, indices=indices(*sync_pts["right_insole"])),
			], dim=1)

	# Compute contact-related information
	insoles["forces"] = Forces.from_pressures(insoles["pressures"], insoles["insoles_size"], subject.weight)
	insoles["contacts"] = Contacts.from_forces(insoles["forces"])
	
	# Save both processed data
	dst.parent.mkdir(parents=True, exist_ok=True)
	torch.save(mocap | insoles, dst)			
	_logger.info("Preprocessing done for sequence {} performed by subject {}.".format(dst.stem, subject))

if __name__ == "__main__":
	logging.basicConfig(level="INFO")
	for subject in Subject.all():
		mocap_stems = set(f.stem for f in subject.mocap_mvnx_files())
		insoles_stems = set(f.stem for f in subject.insoles_txt_files())
		for sequence in mocap_stems & insoles_stems:
			parse_sequence(subject, sequence, subject.preprocessed_file(sequence))