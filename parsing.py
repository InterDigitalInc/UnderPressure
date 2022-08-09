"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# Python
import csv, re
from datetime import datetime
from functools import reduce
from pathlib import Path
from xml.etree import ElementTree as ET

# PyTorch
import torch

class MVNX:
	_MVNX_NAMESPACE = {"mvnx": "http://www.xsens.com/mvn/mvnx"}
	_SUBJECT_TAG = "mvnx:subject"
	_FRAMES_TAG = "mvnx:frames"
	_POSITION_TAG = "mvnx:position"
	_ORIENTATION_TAG = "mvnx:orientation"
	_FRAMERATE_ATTRIB = "frameRate"

	@classmethod
	def parse(cls, src):
		root_idxs = [0]
		def parse_element(frame, tag):
			data = frame.find(tag, cls._MVNX_NAMESPACE)
			return torch.as_tensor([*map(float, data.text.split())], dtype=torch.float)
		def parse_orientation(frame):
			return parse_element(frame, cls._ORIENTATION_TAG).view(-1, 4)
		def parse_position(frame):
			return parse_element(frame, cls._POSITION_TAG).view(-1, 3)
		def load_skeleton(frame):
			skeleton = parse_position(frame)											# J x 3
			return skeleton - skeleton[root_idxs, :]									# J x 3
		def load_frame(frame):
			angles = parse_orientation(frame)											# J x 4
			skeleton = parse_position(frame)											# J x 3
			return angles, skeleton
		def load_motion(frames, skeleton):
			orientations = torch.empty(len(frames), 23, 4)
			trajectory = torch.empty(len(frames), 1, 3)
			for f, frame in enumerate(frames):
				orientations[f], positions = load_frame(frame)
				trajectory[f] = positions[root_idxs]
			return orientations, skeleton, trajectory
		xml_root = ET.parse(src).getroot()
		subject = xml_root.findall(cls._SUBJECT_TAG, cls._MVNX_NAMESPACE)[0]
		frames = subject.find(cls._FRAMES_TAG, cls._MVNX_NAMESPACE)
		framerate = float(subject.attrib[cls._FRAMERATE_ATTRIB])
		angles, skeleton, trajectory = load_motion(frames[3:], load_skeleton(frames[0]))
		return dict(angles=angles, skeleton=skeleton, trajectory=trajectory, framerate=framerate)

class MoticonTXT:
	_FRAMERATES	= [10.0, 25.0, 50.0, 100.0]

	@classmethod
	def _parse_fileheader(cls, input: str):
		# check and remove prefix
		prefix = "# Measurement "
		assert input.startswith(prefix)
		input = input[len(prefix):]

		# parse and return the record start and stop datetimes and insoles size
		start = datetime.strptime(input[:31], "%a, %d %b %Y - %H:%M:%S.%f")
		duration = datetime.strptime(input[32:43], "(%M:%S.%f)") - datetime.strptime("00:00.00", "%M:%S.%f")
		insoles_size = int(input[44:].split(",")[0].split(" ")[-1])
		return start, start + duration, insoles_size

	@classmethod
	def _parse_dataheader(cls, header: str):
		if header == "time":
			return "time", "s"
		
		remainder = header[15:] if header.startswith("External Input") else header
		m = re.fullmatch("(?P<quantity>.[^\[]*)\[(?P<unit>.*)\]", remainder)
		if m is None:
			raise ValueError("Failed to parse header '{}'.".format(header))
		return m["quantity"], m["unit"]
		
	@classmethod
	def _parse_dataheaders(cls, headline: str):
		# check and remove prefix
		prefix = "# "
		assert headline.startswith(prefix)
		headline = headline[len(prefix):]

		# parse headers
		headers = [cls._parse_dataheader(header) for header in headline.split("\t")]
		names = [header[0] for header in headers]
		units = [header[1] for header in headers]
		return names, units
	
	@classmethod
	def _fill_missing_values(cls, data):
		for key, value in data.items():
			if not isinstance(value, torch.Tensor) or not value.isnan().any():
				continue
		
			flat = value.view(value.shape[0], -1) # F x C
			for j in range(flat.shape[1]):
				# set first and last value as first and last valid values to emulate replication padding with interpolation
				valid = flat[:, j].isnan().logical_not()
				flat[[0, -1], j] = flat[valid.nonzero(as_tuple=True)[0][[0, -1]], j]
				
				# interpolate missing values (or ranges)
				def reduce_fn(tail, head):
					if len(tail) > 0 and tail[-1].stop == head.start:
						return tail[:-1] + [range(tail[-1].start, head.stop)]
					else:
						return tail + [head]
				idxs = flat[:, j].isnan().nonzero(as_tuple=True)[0].tolist()
				for r in reduce(reduce_fn, [range(i, i+1) for i in idxs], []):
					weight = torch.arange(1, len(r)+1) / (len(r)+1)
					flat[r.start:r.stop, j] = torch.lerp(flat[[r.start-1], j], flat[[r.stop], j], weight)		
			assert not value.isnan().any()
	
	@classmethod
	def parse(cls, path, fill_missing_values=True):	
		text = Path(path).read_text("utf-8")
		lines = [line for line in text.split("\n")]
		rows = [line for line in lines if len(line) > 0]
		
		# parse headers
		start_datetime, end_datetime, insoles_size = cls._parse_fileheader(next(row for row in rows if row.startswith("# Measurement")))
		quantities, units = cls._parse_dataheaders(next(row for row in rows if row.startswith("# time")))
		
		# parse recorded values
		records = [row.split("\t") for row in rows if not row.startswith("#")]
		assert all(len(record) == len(records[0]) for record in records)
		data = torch.empty(len(records), len(records[0])) # F x Q
		for i in range(len(records)):
			for j in range(len(quantities)):
				data[i, j] = float("nan" if len(records[i][j]) == 0 else records[i][j])
		
		# discard quantities always missing
		valid_indices = data.isnan().logical_not().any(dim=0).nonzero(as_tuple=True)[0]
		data = data[:, valid_indices]
		quantities = [quantities[j] for j in valid_indices]
		units = [units[j] for j in valid_indices]
		
		# extract framerate from timestamps
		time_idx = quantities.index("time")
		timestamps = data[:, time_idx]
		data = torch.cat([data[:, :time_idx], data[:, time_idx+1:]], dim=1)
		quantities.pop(time_idx), units.pop(time_idx)
		framerate = 1.0 / timestamps.diff(dim=0).mean().item()
		framerate = min(((fr, abs(framerate - fr)) for fr in cls._FRAMERATES), key=lambda t: t[1])[0]

		# group quantities into meaningful vectors 
		groups = {
			"force-left": ["left total force"],
			"force-right": ["right total force"],
			"CoP-left": ["left center of pressure {}".format(c) for c in ["X", "Y"]],
			"CoP-right": ["right center of pressure {}".format(c) for c in ["X", "Y"]],
			"acceleration-left": ["left acceleration {}".format(c) for c in ["X", "Y", "Z"]],
			"acceleration-right": ["right acceleration {}".format(c) for c in ["X", "Y", "Z"]],
			"pressures-left": ["left pressure {}".format(c) for c in range(1, 17)],
			"pressures-right": ["right pressure {}".format(c) for c in range(1, 17)],
			"angular-left": ["left angular {}".format(c) for c in ["X", "Y", "Z"]],
			"angular-right": ["right angular {}".format(c) for c in ["X", "Y", "Z"]],
		}
		for q in quantities:
			if not any(q in group for group in groups.values()):
				groups[q] = [q]
		assert all(all(units[quantities.index(q)] == units[quantities.index(group[0])] for q in group) for group in groups.values())
		
		output = {key:torch.stack([data[:, quantities.index(q)] for q in groups[key]], dim=-1) for key in groups}
		units = {key:units[quantities.index(groups[key][0])] for key in groups}

		# group left/right vectors along new dimension
		for key in list(output.keys()):
			if key.endswith("-left"):
				key, lkey, rkey = key.replace("-left", ""), key, key.replace("-left", "-right"), 
				output[key] = torch.stack([output.pop(lkey), output.pop(rkey)], dim=1)
				unit = units.pop(lkey)
				assert unit == units.pop(rkey)
				units[key] = unit
			elif not key.endswith("-right"):
				output[key] = output[key].unsqueeze(1)
		
		# include metadata in output, fill missing values if requested and return
		data = output | dict(framerate=framerate, start_datetime=start_datetime, end_datetime=end_datetime, insoles_size=insoles_size, units=units)
		if fill_missing_values:
			cls._fill_missing_values(data)
		return data

class SyncCSV:
	def parse(src):
		output = {}
		with open(src, newline="") as csvfile:
			for row in csv.DictReader(csvfile):
				output[row.pop("obj")] = (int(row.pop("start")), int(row.pop("stop")))
		return output