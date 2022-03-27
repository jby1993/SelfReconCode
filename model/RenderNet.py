import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.autograd.functional as F
from .Embedder import get_embedder
import utils

class RenderingNetwork_view_norm(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires_n=0,
			multires_v=0
	):
		super().__init__()

		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedv_fn = None
		self.multires_v=multires_v
		if multires_v > 0:
			embedv_fn, input_ch = get_embedder(multires_v)
			self.embedv_fn = embedv_fn
			dims[0] += (input_ch - 3)

		self.embedn_fn = None
		self.multires_n=multires_n
		if multires_n > 0:
			embedn_fn, input_ch = get_embedder(multires_n)
			self.embedn_fn = embedn_fn
			dims[0] += (input_ch - 3)

		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)

			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, points, normals, view_dirs, feature_vectors,ratio):
		ratio=ratio['renderRatio']
		if self.embedv_fn is not None:
			if ratio is None:	#one weight, all equal one
				view_dirs = self.embedv_fn(view_dirs)
			elif ratio<=0: #zero weight
				view_dirs = self.embedv_fn(view_dirs, [0. for _ in range(self.multires_v*2)])
			else:
				view_dirs = self.embedv_fn(view_dirs, utils.annealing_weights(self.multires_v,ratio))
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))

		if self.mode == 'idr':
			rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
		elif self.mode == 'no_view_dir':
			rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
		elif self.mode == 'no_normal':
			rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

		x = rendering_input

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)

		x = self.tanh(x)
		return x

def getRenderNet(device,conf):
	if conf.get_string('type')=='RenderingNetwork_view_norm':
		return RenderingNetwork_view_norm(conf.get_int('condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True, \
				multires_v=conf.get_int('multires_v'),multires_n=conf.get_int('multires_n')).to(device)
	else:
		return globals()[conf.get_string('type')](conf.get_int('condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires=conf.get_int('multires')).to(device)