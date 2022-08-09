"""
	Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
	code is made available under the license found in the LICENSE.txt at the root
	directory of the repository.
"""

# UnderPressure
import anim, util
from data import FRAMERATE, TOPOLOGY, Forces

# Python
import math, time
from pathlib import Path
from tempfile import TemporaryDirectory
from tkinter import Tk

# PyTorch
import torch

# Panda3D
import panda3d.core as pd
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.DirectGui import DGG, DirectButton, DirectLabel, DirectSlider
from direct.gui.OnscreenText import OnscreenText

def rotate(pts, axis, angle):
	axis = torch.nn.functional.normalize(axis, p=2, dim=-1)
	rmat = util.so3.to_SO3(angle * axis)
	return torch.einsum("ij,...j->...i", rmat, pts)

class Geo:
	@classmethod
	def xy2xyz(cls, tensor, z=0.0, dim=-1):													# [...] x 2 x [...]
		assert tensor.shape[dim] == 2
		x, y = tensor.unbind(dim)															# [...] x	 [...]
		return torch.stack([x, y, torch.full_like(x, z)], dim)								# [...] x 3 x [...]

class GeomBuilder:
	def __init__(self, name, color=None):
		self.name = name
		self.vdata = pd.GeomVertexData(name, pd.GeomVertexFormat.getV3n3c4(), pd.Geom.UHStatic)
		self.vertexWriter = pd.GeomVertexWriter(self.vdata, "vertex")
		self.normalWriter = pd.GeomVertexWriter(self.vdata, "normal")
		self.colorWriter = pd.GeomVertexWriter(self.vdata, "color")
		
		self.triangles = pd.GeomTriangles(pd.Geom.UHStatic)
		self.currentRow = 0
		self.color = color

	def _vertex(self, point, normal, prim=None, color=None):
		color = color or self.color or [1, 1, 1, 1]
		self.vertexWriter.addData3(*point)
		self.normalWriter.addData3(*normal)
		self.colorWriter.addData4(color[0], color[1], color[2], color[3])
		if prim is not None:
			prim.addVertex(self.currentRow)
		self.currentRow += 1

	def triangle(self, p1, p2, p3, color=None, normal=None):
		if normal is None:
			normal = (p2 - p1).cross(p3 - p1).normalized()
		self._vertex(p1, normal, self.triangles, color)
		self._vertex(p2, normal, self.triangles, color)
		self._vertex(p3, normal, self.triangles, color)
		self.triangles.closePrimitive()

	def polygon(self, pts, color=None):
		triangulator = pd.Triangulator3()
		for pt in pts:
			triangulator.addPolygonVertex(triangulator.addVertex(*pt))
		triangulator.triangulate()
		for i in range(triangulator.getNumTriangles()):
			self.triangle(
				triangulator.getVertex(triangulator.getTriangleV0(i)),
				triangulator.getVertex(triangulator.getTriangleV1(i)),
				triangulator.getVertex(triangulator.getTriangleV2(i)),
				color=color
			)

	def thick_polygon(self, pts, tickness, color=None, face_color=None, edge_color=None):
		pts = torch.as_tensor(pts)
		bottom, top = pts, pts + torch.as_tensor([0, 0, tickness])
		self.polygon(top, face_color or color)
		self.polygon(bottom, face_color or color)
		for i in range(0, top.shape[0]):
			j = (i + 1) % top.shape[0]
			self.polygon([top[i], top[j], bottom[j], bottom[i]], edge_color or color)

	def cylinder(self, p1, p2, radius, n=24, color=None):
		p1, p2 = torch.as_tensor(p1), torch.as_tensor(p2)
		axis = torch.nn.functional.normalize(p2 - p1, p=2, dim=-1)					# 3
		
		ex = torch.as_tensor([1.0, 0.0, 0.0])
		if (axis != ex).any():
			rvec = radius * torch.nn.functional.normalize(ex.cross(axis), p=2, dim=-1)
		else:
			rvec = torch.as_tensor([0.0, radius, 0.0])
		
		for i in range(n):
			self.triangle(
				p1 + rotate(rvec, axis, i / n * math.tau),
				p2 + rotate(rvec, axis, (i + 0.5) / n * math.tau),
				p2 + rotate(rvec, axis, (i - 0.5) / n * math.tau),
				color=color,
				normal=rotate(rvec, axis, i / n * math.tau),
			)
			self.triangle(
				p2 + rotate(rvec, axis, (i + 0.5) / n * math.tau),
				p1 + rotate(rvec, axis, (i + 1) / n * math.tau),
				p1 + rotate(rvec, axis, i / n * math.tau),
				color=color,
				normal=rotate(rvec, axis, (i + 0.5) / n * math.tau),
			)

	def cube(self, center, side, color=None):
		return self.box(center, [side, side, side], color=color)

	def box(self, center, sides, color=None):
		center = torch.as_tensor(center)
		sx, sy, sz = sides[0] / 2, sides[1] / 2, sides[2] / 2
		vertices = torch.stack([
			center + torch.as_tensor([-sx, -sy, -sz]),
			center + torch.as_tensor([-sx, sy, -sz]),
			center + torch.as_tensor([-sx, sy, sz]),
			center + torch.as_tensor([-sx, -sy, sz]),
			center + torch.as_tensor([sx, -sy, -sz]),
			center + torch.as_tensor([sx, sy, -sz]),
			center + torch.as_tensor([sx, sy, sz]),
			center + torch.as_tensor([sx, -sy, sz]),
		])
		self.polygon(vertices[[0, 1, 2, 3]], color)
		self.polygon(vertices[[0, 4, 5, 1]], color)
		self.polygon(vertices[[1, 5, 6, 2]], color)
		self.polygon(vertices[[2, 6, 7, 3]], color)
		self.polygon(vertices[[3, 7, 4, 0]], color)
		self.polygon(vertices[[7, 6, 5, 4]], color)

	def build(self):
		geom = pd.Geom(self.vdata)
		geom.addPrimitive(self.triangles)
		node = pd.GeomNode(self.name)
		node.addGeom(geom)
		return node

class AnimationLoop:
	def __init__(self, length: int, fps: float, start: bool = True):
		self._length, self._fps = int(length), float(fps)
		self._time, self._frame = 0.0, 0.0
		self._stop, self._running = False, bool(start)
		self.functions = []

	@property
	def length(self) -> int:
		return self._length
	@length.setter
	def length(self, length: int):
		self._length = int(length)
	def __len__(self) -> int:
		return self.length

	@property
	def fps(self) -> float:
		return self._fps
	@fps.setter
	def fps(self, value: float):
		self._fps = float(value)

	@property
	def frame(self) -> float:
		return self._frame % self.length
	@property
	def running(self) -> bool:
		return self._running

	def reset(self):
		self._time = 0.0
		self._frame = 0.0
	def pause(self):
		self._running = False
	def unpause(self):
		self._running = True
	def stop(self):
		self._stop = True

	def bind(self, fn):
		self.functions.append(fn)
	def unbind(self, fn):
		self.functions.remove(fn)

	def __call__(self, state):
		if self._stop:
			return Task.done
		if self._running:
			self._frame += (state.time - self._time) * self.fps
			for fn in self.functions:
				fn(self.frame)
		self._frame %= self.length
		self._time = state.time
		return Task.cont

class Animatable:
	def __init__(self, loop: AnimationLoop):
		self.loop = loop
		
	@property
	def loop(self):
		return self._loop
		
	@loop.setter
	def loop(self, loop: AnimationLoop):
		del self.loop
		assert isinstance(loop, AnimationLoop)
		self._loop = loop
		self._loop.bind(self.update)

	@loop.deleter
	def loop(self):
		if hasattr(self, "_loop"):
			self._loop.unbind(self.update)
			del self._loop

	@property
	def nframes(self):
		return len(self.loop)
	
	def update(self, frame: int):
		prev_frame, next_frame = int(math.floor(frame)), min(int(math.ceil(frame)), self.nframes-1)
		dframe = frame - prev_frame
		self.__update__(frame, prev_frame, next_frame, dframe)
		
	def __update__(self, frame: int, prev_frame: int, next_frame: int, dframe: float):
		raise NotImplementedError()

class Ground(Animatable):
	DEFAULT_COLOR1 = 3 * (0.4, ) + (1.0, )
	DEFAULT_COLOR2 = 3 * (0.6, ) + (1.0, )
	
	def __init__(self, parent, grid, size, origin=(0, 0), color1=DEFAULT_COLOR1, color2=DEFAULT_COLOR2, loop=None, trajectory=None):
		grid, size = torch.as_tensor(grid).int(), torch.as_tensor(size)
		origin = torch.as_tensor(origin)
		cell = size / grid
		builder = GeomBuilder("ground")
		
		for x in range(grid[0]):
			for y in range(grid[1]):
				vertices = origin - 0.5*size + torch.stack([
					torch.as_tensor([x, y]) * cell,
					torch.as_tensor([x, y+1]) * cell,
					torch.as_tensor([x+1, y+1]) * cell,
					torch.as_tensor([x+1, y]) * cell,
				])
				color = color1 if x%2 == y%2 else color2
				builder.polygon(Geo.xy2xyz(vertices), color)
		self._node = parent.attachNewNode(builder.build())
		
		if loop is not None:
			super().__init__(loop)
		if trajectory is not None:
			self.trajectory = trajectory
		
	@property
	def trajectory(self):
		return self._trajectory.clone()
	@trajectory.setter
	def trajectory(self, trajectory):
		self._trajectory = trajectory.clone()
			
	def __update__(self, frame: int, prev_frame: int, next_frame: int, dframe: float):
		position = torch.lerp(*self.trajectory[[prev_frame, next_frame]], dframe)
		self._node.setPos(*position.tolist())

class Insole(Animatable):
	# geometry
	INSOLE_VERTICES = torch.load(Path(__file__).parent / "geo_insoles.pth")
	INSOLE_VERTICES_BBOX = torch.stack([INSOLE_VERTICES.amin(dim=0), INSOLE_VERTICES.amax(dim=0)])
	INSOLE_VERTICES_LENGTH = (INSOLE_VERTICES_BBOX[1, 1] - INSOLE_VERTICES_BBOX[0, 1])
	INSOLE_VERTICES_CENTER = 0.5 * (INSOLE_VERTICES_BBOX[0] + INSOLE_VERTICES_BBOX[1])
	CELLS_VERTICES = torch.load(Path(__file__).parent / "geo_insole_cells.pth")

	# colors
	INSOLE_COLOR = (0.2, 0.2, 0.2, 1.0)
	TOE_COLOR = (150/255, 48/255, 48/255, 1.0)
	ARCH_COLOR = (0.5, 0.5, 0.5, 1.0)
	HEEL_COLOR = (48/255, 84/255, 150/255, 1.0)

	def __init__(self, parent, side, origin, length, angle, zscale=0.025, loop=None, forces=None):
		self._values = torch.zeros(16)
		self._side = side
		self.__init_geometry__(parent, side, origin, length, angle)
		self._zscale = float(zscale)
		if loop is not None:
			super().__init__(loop)
		if forces is not None:
			self.forces = forces

	@classmethod
	def left(cls, parent, *args, **kwargs):
		return cls(parent, "left", *args, **kwargs)
		
	@classmethod
	def right(cls, parent, *args, **kwargs):
		return cls(parent, "right", *args, **kwargs)

	def is_left(self) -> bool:
		return self._side == "left"
		
	def is_right(self) -> bool:
		return self._side == "right"

	def __init_geometry__(self, parent, side, origin, length, angle):
		self._origin, length = torch.as_tensor(origin), torch.as_tensor(length)
		angle = (180 + torch.as_tensor(angle)) / 180 * math.pi
		side = {"right": 1, "left": -1}[side]
		
		# prepare global transform
		transform_mat = torch.as_tensor([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]) # rotation
		transform_mat = torch.matmul(transform_mat, (length / self.INSOLE_VERTICES_LENGTH).expand(2).diag()) # scale
		transform_mat = torch.matmul(transform_mat, torch.as_tensor([side, 1.0]).diag()) # left
		transform = lambda pts: Geo.xy2xyz(torch.einsum("ij,nj->ni", transform_mat, pts.clone() - self.INSOLE_VERTICES_CENTER))
		
		self._base_tickness = 0.002
		
		# insole
		builders = {}
		insole_vertices = transform(self.INSOLE_VERTICES)
		if side == -1:
			insole_vertices = insole_vertices.flip(0)
		builders["insole"] = GeomBuilder("insole")
		builders["insole"].thick_polygon(
			self._origin + insole_vertices, tickness=self._base_tickness,
			face_color=self.INSOLE_COLOR, edge_color=self.INSOLE_COLOR
		)
		
		# cells
		for i, cell_vertices in enumerate(self.CELLS_VERTICES):
			cell_vertices = transform(cell_vertices)
			if side == -1:
				cell_vertices = cell_vertices.flip(0)
			cell_name = "cell{}".format(i)
			cell_face_color = self.TOE_COLOR if i in Forces.FRONT_CELLS else self.HEEL_COLOR if i in Forces.BACK_CELLS else self.ARCH_COLOR
			builders[cell_name] = GeomBuilder(cell_name)
			builders[cell_name].thick_polygon(
				cell_vertices, tickness=1.0,
				face_color=cell_face_color, edge_color=cell_face_color
			)
			
		self._nodes = {key: parent.attachNewNode(builder.build()) for key, builder in builders.items()}		
		cells_origin = self._origin + torch.as_tensor([0.0, 0.0, self._base_tickness])
		for i in range(len(self.CELLS_VERTICES)):		
			self._nodes["cell{}".format(i)].setPos(*cells_origin.tolist())
	
	def __getitem__(self, index):
		return self._values.__getitem__(index)
	def __setitem__(self, index, value):
		self._values.__setitem__(index, value)
		for i in range(16):
			z = self._zscale * self._values[i].item() + 1e-5
			self._nodes["cell{}".format(i)].setSz(z)

	@property
	def forces(self):
		return self._forces.clone()
	@forces.setter
	def forces(self, forces):
		self._forces = forces.clone()
		
	def __update__(self, frame: int, prev_frame: int, next_frame: int, dframe: float):
		self[:] = torch.lerp(*self.forces[[prev_frame, next_frame]], dframe)		

class InsolesPair:
	def __init__(self, left, right, loop=None, forces=None):
		self.left, self.right = left, right
		if loop is not None:
			self.left.loop = loop
			self.right.loop = loop
		if forces is not None:
			self.forces = forces

	@property
	def left(self):
		return self._left
	@left.setter
	def left(self, left):
		assert isinstance(left, Insole) and left.is_left()
		self._left = left
		
	@property
	def right(self):
		return self._right
	@right.setter
	def right(self, right):
		assert isinstance(right, Insole) and right.is_right()
		self._right = right

	def __setitem__(self, index, value):
		values = self[:]
		values.__setitem__(index, value)
		self.left[:] = values[0]
		self.right[:] = values[1]

	def __getitem__(self, index):
		return torch.stack([self.left[:], self.right[:]]).__getitem__(index)
	
	@property
	def forces(self):
		return self.left.forces, self.right.forces
	@forces.setter
	def forces(self, forces):
		self.left.forces = forces[0]
		self.right.forces = forces[1]
	
	def update(self, frame):
		self.left.update(frame)
		self.right.update(frame)

class Skeleton(Animatable):
	def __init__(self, parent, skeleton, loop=None, angles=None, trajectory=None, label=""):
		self._skeleton = skeleton
		self._joints = []
		for joint in TOPOLOGY:
			node = GeomBuilder(joint)
			node.cube((0, 0, 0), side=0.05, color=[0.2, 0.2, 0.2, 0.0])
			self._joints.append(parent.attachNewNode(node.build()))
		if loop is not None:
			super().__init__(loop)
		if angles is not None:
			self.angles = angles
		if trajectory is not None:
			self.trajectory = trajectory	
		if label:
			self._label = pd.TextNode(label)
			self._label.setText(label)
			self._label.setAlign(pd.TextNode.ACenter)
			self._label = parent.attachNewNode(self._label)
			self._label.setScale(0.1)
			self._label.setPos(0.0, 0.0, 1.8)
		
	@property
	def angles(self):
		return self._angles.clone()
	@angles.setter
	def angles(self, angles):
		self._angles = angles.clone()
		
	@property
	def trajectory(self):
		return self._trajectory.clone()
	@trajectory.setter
	def trajectory(self, trajectory):
		self._trajectory = trajectory.clone()
		self._label_traj = util.gma(self.trajectory, size=50, std=50/3, dim=0)
	
	def __update__(self, frame: int, prev_frame: int, next_frame: int, dframe: float):
		angles = util.SU2.slerp(*self.angles[[prev_frame, next_frame]], dframe)
		positions = anim.FK(angles, self._skeleton, 0, TOPOLOGY)
		if hasattr(self, "_trajectory"):
			positions += torch.lerp(*self.trajectory[[prev_frame, next_frame]], dframe)		
		if hasattr(self, "_label") and hasattr(self, "_label_traj"):
			self._label.setPos(*torch.lerp(*self._label_traj[[prev_frame, next_frame]], dframe)[:2].tolist(), 1.8)				
		for j, joint in enumerate(self._joints):
			joint.setQuat(pd.LQuaternion(*angles[j].tolist()))
			joint.setPos(*positions[j].tolist())
	
class MocapAndvGRFsApp(ShowBase):
	def __init__(self, motions: list, vGRFs: list = [], motion_labels=None, vGRF_labels=None):
		motion_labels = [None for _ in motions] if motion_labels is None else motion_labels
		vGRF_labels = [None for _ in vGRFs] if vGRF_labels is None else vGRF_labels
		
		margin = 100
		screen = Tk().winfo_screenwidth() - margin, Tk().winfo_screenheight() - margin
		width, height = (screen[0], int(screen[0] / 1.8)) if screen[0] / screen[1] < 1.8 else (int(screen[1] * 1.8), screen[1]) 
		pd.loadPrcFileData("", "win-origin {} {}".format((screen[0] + margin - width) // 2, (screen[1] + margin - height) // 2))
		pd.loadPrcFileData("", "win-size {} {}".format(width, height))
		pd.loadPrcFileData("", "win-fixed-size 1")
		
		super().__init__()
		self.render.setTwoSided(True)
		self.disableMouse()
		self.__init_lights__()
		self.__init_camera__(fov=50, near=0.01, far=50, pos=[0.0, -7.0, 3.0], dir=[0.0, 1.2, -0.5])
		
		# init animation loop	
		assert len({v.shape[0] for v in vGRFs} | {m[0].shape[0] for m in motions}) == 1		
		nframes = motions[0][0].shape[0]
		self.animation_loop = AnimationLoop(length=nframes, fps=FRAMERATE, start=True)
		self.taskMgr.add(self.animation_loop, "animation_loop")

		# insoles
		if len(vGRFs) > 0:
			vGRFs_scale = 1 / torch.stack(list(vGRFs)).max()	
			for index, (value, label) in enumerate(zip(vGRFs, vGRF_labels)):
				x = index * 0.111 - 0.111 * (len(vGRFs) - 1) / 2
				self.add_insoles([
					x, self.cam.getPos()[1] + 0.60 + 0.08, self.cam.getPos()[2] - 0.435],
					vGRFs_scale * value, plate=True, label=label,
				)

		# get global trajectory
		trajectories = torch.cat([motion[2] for motion in motions], dim=-2)
		global_trajectory = trajectories.mean(dim=-2)
		global_trajectory = util.gma(global_trajectory, size=30, std=30/3, dim=0)
		global_trajectory[..., 2] = 0.0
				
		c = 0.25
		global_bbox = trajectories.amax(dim=(0, 1)) - trajectories.amin(dim=(0, 1))
		global_position = global_trajectory.mean(dim=0)
		ground_grid = ((global_bbox[:2] + 3) / c).ceil()
		self._ground = Ground(
			self.render, grid=ground_grid, size=c*ground_grid, origin=global_position[:2],
			loop=self.animation_loop,
			trajectory=-global_trajectory,
		)
		
		for motion, label in zip(motions, motion_labels):
			angles, skeleton, trajectory = motion[0], motion[1].reshape(-1, 3), motion[2].reshape(-1, 3)
			trajectory = trajectory - global_trajectory
			self._skeleton = Skeleton(
				self.render, skeleton,
				loop=self.animation_loop, angles=angles, trajectory=trajectory,
				label=label,
			)
		
		# Framerate buttons
		button025 = DirectButton(text="0.25x", scale=0.05, pos=(-0.2, 0.0, 0.96), command=lambda: self.change_framerate(0.25))
		button050 = DirectButton(text="0.50x", scale=0.05, pos=(0.0, 0.0, 0.96), command=lambda: self.change_framerate(0.50))
		button100 = DirectButton(text="1.00x", scale=0.05, pos=(0.2, 0.0, 0.96), command=lambda: self.change_framerate(1.00))
	
	def change_framerate(self, rate):
		self.animation_loop.fps = rate * FRAMERATE
		
	def __init_lights__(self):
		def point_light(position, color, name=""):
			light = pd.PointLight(name)
			light.setColor(color)
			light = self.render.attachNewNode(light)
			light.setPos(*position)
			self.render.setLight(light)
		
		intensity = 0.6
		point_light((5, 5, -5), (intensity, intensity, intensity, 1))
		point_light((0, 5, -5), (intensity, intensity, intensity, 1))
		point_light((5, 0, -5), (intensity, intensity, intensity, 1))
		
	def __init_camera__(self, **kwargs):
		if "near" in kwargs:
			self.camLens.setNear(kwargs["near"])
		if "far" in kwargs:
			self.camLens.setFar(kwargs["far"])		
		if "fov" in kwargs:
			self.camLens.setFov(kwargs["fov"])
		if "pos" in kwargs:
			self.cam.setPos(*kwargs["pos"])
		if "target" in kwargs:
			pos, target = torch.as_tensor(self.cam.getPos()), torch.as_tensor(kwargs["target"])
			dir = (target - pos).tolist()
		elif "dir" in kwargs:
			dir = kwargs["dir"]
		self.camLens.setViewVector(
			*dir,
			*kwargs.get("up", [0.0, 0.0, 1.0]),
		)

	def add_insoles(self, origin, forces, plate=False, label=None):
		insoles = InsolesPair(
			Insole.left(self.render, origin=[origin[0] - 0.021, origin[1], origin[2]], length=0.15, angle=10.0),
			Insole.right(self.render, origin=[origin[0] + 0.021, origin[1], origin[2]], length=0.15, angle=-10.0),
			loop=self.animation_loop, forces=forces.unbind(dim=-2),
		)
		
		width, depth, thickness = 0.108, 0.16, 0.01
		if plate:
			builder = GeomBuilder("insoles_plate")
			builder.box([origin[0], origin[1], origin[2] - thickness/2], [width, depth, thickness], color=(0.8, 0.8, 0.8, 1.0))
			self.render.attachNewNode(builder.build())
		
		if isinstance(label, str):
			text = pd.TextNode(label)
			text.setText(label)
			text.setAlign(pd.TextNode.ACenter)
			text = self.render.attachNewNode(text)
			text.setScale(thickness)
			text.setPos(origin[0], origin[1] - depth/2 - 1e-4, origin[2] - 0.8*thickness)		

if __name__ == "__main__":
	import models
	from data import Dataset
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("-subj", "-subject", default="S9", help="Subject to be selected; default: S9")
	parser.add_argument("-seq", "-sequence", default="WalkRandomSlow", help="Sequence to be selected; default: WalkRandomSlow")
	parser.add_argument("-checkpoint", default="pretrained.tar")
	args = parser.parse_args()
	
	dataset = Dataset("{}-{}".format(args.subj, args.seq))
	item = dataset[0]
	angles, skeleton, trajectory = item["angles"], item["skeleton"], item["trajectory"]
	vGRFs_gt = item["forces"]
	positions = anim.FK(angles, skeleton, trajectory, TOPOLOGY)
	
	checkpoint = torch.load(args.checkpoint)
	model = models.DeepNetwork(state_dict=checkpoint["model"]).eval()
	with torch.no_grad():
		vGRFs_pred = model.vGRFs(positions.unsqueeze(0)).squeeze(0)
	vGRFs_abs_error = (vGRFs_gt - vGRFs_pred).abs()

	MocapAndvGRFsApp(
		[(angles, skeleton, trajectory)],
		[vGRFs_gt, vGRFs_pred, vGRFs_abs_error],
		vGRF_labels=["Ground Truth", "Estimated", "Abs. Error"],
	).run()

