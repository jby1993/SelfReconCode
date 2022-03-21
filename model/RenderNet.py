import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.autograd.functional as F
from .Embedder import get_embedder
import utils
class RenderingNetwork(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires=0
	):
		super().__init__()

		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedview_fn = None
		self.multires=multires
		if multires > 0:
			embedview_fn, input_ch = get_embedder(multires)
			self.embedview_fn = embedview_fn
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
		if self.embedview_fn is not None:
			if ratio is None:	#one weight, all equal one
				view_dirs = self.embedview_fn(view_dirs)
			elif ratio<=0: #zero weight
				view_dirs = self.embedview_fn(view_dirs, [0. for _ in range(self.multires*2)])
			else:
				view_dirs = self.embedview_fn(view_dirs, utils.annealing_weights(self.multires,ratio))

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


class RenderingNetwork_pos(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires=0
	):
		super().__init__()

		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedview_fn = None
		self.multires=multires
		if multires > 0:
			embedview_fn, input_ch = get_embedder(multires)
			self.embedview_fn = embedview_fn
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
		if self.embedview_fn is not None:
			if ratio is None:	#one weight, all equal one
				points = self.embedview_fn(points)
			elif ratio<=0: #zero weight
				points = self.embedview_fn(points, [0. for _ in range(self.multires*2)])
			else:
				points = self.embedview_fn(points, utils.annealing_weights(self.multires,ratio))

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

class RenderingNetwork_pos_norm(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires=0
	):
		super().__init__()

		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedview_fn = None
		self.multires=multires
		if multires > 0:
			embedview_fn, input_ch = get_embedder(multires)
			self.embedview_fn = embedview_fn
			dims[0] += (input_ch - 3)*2

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
		if self.embedview_fn is not None:
			if ratio is None:	#one weight, all equal one
				points = self.embedview_fn(points)
				normals = self.embedview_fn(normals)
			elif ratio<=0: #zero weight
				points = self.embedview_fn(points, [0. for _ in range(self.multires*2)])
				normals = self.embedview_fn(normals, [0. for _ in range(self.multires*2)])
			else:
				points = self.embedview_fn(points, utils.annealing_weights(self.multires,ratio))
				normals = self.embedview_fn(normals, utils.annealing_weights(self.multires,ratio))

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


class RenderingNetwork_pos_norm2(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires_p=0,
			multires_n=0
	):
		super().__init__()

		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedp_fn = None
		self.multires_p=multires_p
		if multires_p > 0:
			embedp_fn, input_ch = get_embedder(multires_p)
			self.embedp_fn = embedp_fn
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
		if self.embedp_fn is not None:
			if ratio is None:	#one weight, all equal one
				points = self.embedp_fn(points)
			elif ratio<=0: #zero weight
				points = self.embedp_fn(points, [0. for _ in range(self.multires_p*2)])
			else:
				points = self.embedp_fn(points, utils.annealing_weights(self.multires_p,ratio))
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


class RenderingNetwork_IDRAug(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires_p=0,
			multires_x=0,
			multires_v=0,
			multires_n=0
	):
		super().__init__()
		self.enable_px=True
		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedp_fn = None
		self.multires_p=multires_p
		if multires_p > 0:
			embedp_fn, input_ch = get_embedder(multires_p)
			self.embedp_fn = embedp_fn
			dims[0] += (input_ch - 3)

		self.embedx_fn = None
		self.multires_x=multires_x
		if multires_x > 0:
			embedx_fn, input_ch = get_embedder(multires_x)
			self.embedx_fn = embedx_fn
			dims[0] += (input_ch - 3)

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

	def forward(self, points,xs, normals, view_dirs, feature_vectors,ratio):
		ratio=ratio['renderRatio']
		if self.embedp_fn is not None:
			if ratio is None:	#one weight, all equal one
				pointss = self.embedp_fn(points)
			elif ratio<=0: #zero weight
				points = self.embedp_fn(points, [0. for _ in range(self.multires_p*2)])
			else:
				points = self.embedp_fn(points, utils.annealing_weights(self.multires_p,ratio))
		if self.embedx_fn is not None:
			if ratio is None:	#one weight, all equal one
				xs = self.embedx_fn(xs)
			elif ratio<=0: #zero weight
				xs = self.embedx_fn(xs, [0. for _ in range(self.multires_x*2)])
			else:
				xs = self.embedx_fn(xs, utils.annealing_weights(self.multires_x,ratio))
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))

		if self.embedv_fn is not None:
			if ratio is None:	#one weight, all equal one
				view_dirs = self.embedv_fn(view_dirs)
			elif ratio<=0: #zero weight
				view_dirs = self.embedv_fn(view_dirs, [0. for _ in range(self.multires_v*2)])
			else:
				view_dirs = self.embedv_fn(view_dirs, utils.annealing_weights(self.multires_v,ratio))

		rendering_input = torch.cat([points, xs, view_dirs, normals, feature_vectors], dim=-1)

		x = rendering_input

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)

		x = self.tanh(x)
		return x


class FeatureTransformer(nn.Module):
	def __init__(self,feature_vector_size,frame_vector_size,weight_norm=True):
		super().__init__()
		dims=[feature_vector_size+frame_vector_size,feature_vector_size+frame_vector_size,feature_vector_size,feature_vector_size]
		self.num_layers = len(dims)
		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.softplus = nn.Softplus(beta=100)
	def forward(self,feature_vectors,frame_vectors):
		x = torch.cat([feature_vectors,frame_vectors],dim=-1)

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)
			if l < self.num_layers - 2:
				x = self.softplus(x)
		return x

class RenderingNetwork_IDRAug_framecond(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			frame_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires_x=0,
			multires_v=0,
			multires_n=0
	):
		super().__init__()
		self.featureTrans=FeatureTransformer(feature_vector_size,frame_vector_size,weight_norm)
		self.enable_px=True
		self.enable_framefeature=True	
		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedx_fn = None
		self.multires_x=multires_x
		if multires_x > 0:
			embedx_fn, input_ch = get_embedder(multires_x)
			self.embedx_fn = embedx_fn
			dims[0] += (input_ch - 3)

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

	def forward(self, points,xs, normals, view_dirs, feature_vectors,frame_vectorss,ratio):
		ratio=ratio['renderRatio']
		if self.embedx_fn is not None:
			if ratio is None:	#one weight, all equal one
				xs = self.embedx_fn(xs)
			elif ratio<=0: #zero weight
				xs = self.embedx_fn(xs, [0. for _ in range(self.multires_x*2)])
			else:
				xs = self.embedx_fn(xs, utils.annealing_weights(self.multires_x,ratio))
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))

		if self.embedv_fn is not None:
			if ratio is None:	#one weight, all equal one
				view_dirs = self.embedv_fn(view_dirs)
			elif ratio<=0: #zero weight
				view_dirs = self.embedv_fn(view_dirs, [0. for _ in range(self.multires_v*2)])
			else:
				view_dirs = self.embedv_fn(view_dirs, utils.annealing_weights(self.multires_v,ratio))

		feature_vectors=self.featureTrans(feature_vectors,frame_vectorss)

		rendering_input = torch.cat([points, xs, view_dirs, normals, feature_vectors], dim=-1)

		x = rendering_input

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)

		x = self.tanh(x)
		return x

class albedoNet(nn.Module):
	def __init__(self,weight_norm=True,multires=0):
		super().__init__()
		dims=[3,512,512,512,512,3]
		self.multires=multires
		self.embed_fn=None
		if multires > 0:
			embed_fn, input_ch = get_embedder(multires)
			self.embed_fn = embed_fn
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
	def forward(self, points,ratio):
		if self.embed_fn is not None:
			if ratio is None:	#one weight, all equal one
				points = self.embed_fn(points)
			elif ratio<=0: #zero weight
				points = self.embed_fn(points, [0. for _ in range(self.multires*2)])
			else:
				points = self.embed_fn(points, utils.annealing_weights(self.multires,ratio))

		x = points

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)

		x = self.tanh(x)
		return x



class RenderingNetwork_IDRAlbedo(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires_p=0,
			multires_x=0,
			multires_v=0,
			multires_n=0
	):
		super().__init__()
		self.albedo=albedoNet(weight_norm,multires_p)
		self.enable_px=True
		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedx_fn = None
		self.multires_x=multires_x
		if multires_x > 0:
			embedx_fn, input_ch = get_embedder(multires_x)
			self.embedx_fn = embedx_fn
			dims[0] += (input_ch - 3)

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
		self.softplus = nn.Softplus()

	def forward(self, points,xs, normals, view_dirs, feature_vectors,ratio):
		albedos=self.albedo(points,1.)
		ratio=ratio['renderRatio']
		if self.embedx_fn is not None:
			if ratio is None:	#one weight, all equal one
				xs = self.embedx_fn(xs)
			elif ratio<=0: #zero weight
				xs = self.embedx_fn(xs, [0. for _ in range(self.multires_x*2)])
			else:
				xs = self.embedx_fn(xs, utils.annealing_weights(self.multires_x,ratio))
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))

		if self.embedv_fn is not None:
			if ratio is None:	#one weight, all equal one
				view_dirs = self.embedv_fn(view_dirs)
			elif ratio<=0: #zero weight
				view_dirs = self.embedv_fn(view_dirs, [0. for _ in range(self.multires_v*2)])
			else:
				view_dirs = self.embedv_fn(view_dirs, utils.annealing_weights(self.multires_v,ratio))

		rendering_input = torch.cat([xs, view_dirs, normals, feature_vectors], dim=-1)

		x = rendering_input

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)

		x = self.softplus(x)
		return (albedos+1.)*x-1.	#note albedo and render net color range(-1.,1.)


class RenderingNetwork_IDRAlbedo_framecond(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			frame_vector_size,
			mode,
			d_in,
			d_out,
			dims,
			weight_norm=True,
			multires_p=0,
			multires_x=0,
			multires_v=0,
			multires_n=0
	):
		super().__init__()
		self.featureTrans=FeatureTransformer(feature_vector_size,frame_vector_size,weight_norm)
		self.albedo=albedoNet(weight_norm,multires_p)
		self.enable_px=True
		self.enable_framefeature=True	
		self.mode = mode
		dims = [d_in + feature_vector_size] + dims + [d_out]

		self.embedx_fn = None
		self.multires_x=multires_x
		if multires_x > 0:
			embedx_fn, input_ch = get_embedder(multires_x)
			self.embedx_fn = embedx_fn
			dims[0] += (input_ch - 3)

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
		self.softplus = nn.Softplus()

	def forward(self, points,xs, normals, view_dirs, feature_vectors,frame_vectors,ratio):
		albedos=self.albedo(points,1.)
		ratio=ratio['renderRatio']
		if self.embedx_fn is not None:
			if ratio is None:	#one weight, all equal one
				xs = self.embedx_fn(xs)
			elif ratio<=0: #zero weight
				xs = self.embedx_fn(xs, [0. for _ in range(self.multires_x*2)])
			else:
				xs = self.embedx_fn(xs, utils.annealing_weights(self.multires_x,ratio))
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))

		if self.embedv_fn is not None:
			if ratio is None:	#one weight, all equal one
				view_dirs = self.embedv_fn(view_dirs)
			elif ratio<=0: #zero weight
				view_dirs = self.embedv_fn(view_dirs, [0. for _ in range(self.multires_v*2)])
			else:
				view_dirs = self.embedv_fn(view_dirs, utils.annealing_weights(self.multires_v,ratio))

		feature_vectors=self.featureTrans(feature_vectors,frame_vectors)
		rendering_input = torch.cat([xs, view_dirs, normals, feature_vectors], dim=-1)

		x = rendering_input

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)

		x = self.softplus(x)
		return (albedos+1.)*x-1.	#note albedo and render net color range(-1.,1.)


class RenderingNetwork_albedo(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			weight_norm=True,
			multires_p=0,
			multires_n=0
	):
		super().__init__()
		self.albedo=albedoNet(weight_norm,multires_p)
		

		dims=[3+3+3+feature_vector_size,256,512,256,3]

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
		self.sigmoid = nn.Sigmoid()


	def forward(self, points, normals, view_dirs, feature_vectors,ratio):
		ratio=ratio['renderRatio']		
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))
		albedos=self.albedo(points,1.)
		
		x = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)


		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)

		x = self.sigmoid(x)

		return (albedos+1.)*x-1.	#note albedo and render net color range(-1.,1.)

class LightNet(nn.Module):
	def __init__(self,out_dim,weight_norm=True):
		super().__init__()
		dims=[3,128,256,128,out_dim]
		
		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, points):
		x = points

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)
		x = self.tanh(x)
		return 2.*x

class SPHLightNet(nn.Module):
	def __init__(self,out_dim,feature_vector_size=0,weight_norm=True):
		super().__init__()
		self.out_dim=out_dim
		dims=[3+feature_vector_size,512,512,512,3*out_dim]
		
		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()

	def forward(self, points, zs=None):
		if zs:
			x = torch.cat([points,zs],dim=-1)
		else:
			x = points

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)
		return x.reshape(-1,self.out_dim,3)

class SPHDCTLightNet(nn.Module):
	def __init__(self,out_dim,feature_vector_size=0,weight_norm=True):
		super().__init__()
		self.out_dim=out_dim
		self.dct_dim=20
		dims=[3+feature_vector_size,512,512,512,3*self.dct_dim]
		self.register_buffer('dctbasis',utils.DCTSpace(self.dct_dim,out_dim))
		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()

	def forward(self, points, zs=None):
		if zs is not None:
			x = torch.cat([points,zs],dim=-1)
		else:
			x = points

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)
		return torch.cat([tx.matmul(self.dctbasis).unsqueeze(-1) for tx in torch.split(x,self.dct_dim,dim=-1)],dim=-1)

class LightCondNet(nn.Module):
	def __init__(self,out_dim,feature_dim,weight_norm=False):
		super().__init__()
		dims=[3+feature_dim,256,512,256,out_dim*3]
		
		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)

			if weight_norm:
				# lin = nn.utils.weight_norm(lin)
				print('LightCondNet:weight norm can influence weight initialization, can not produce small weights as initialization. Now do not use weight_norm')
			# initialize with zeros translation
			if l==self.num_layers-2:
				torch.nn.init.normal_(lin.weight, mean=-5., std=0.001)
				torch.nn.init.constant_(lin.bias, 0.)
			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()
		# self.sigmoid=nn.Sigmoid()
	def forward(self, points, zs):
		x = torch.cat([points,zs],dim=-1)

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)
		# x = self.sigmoid(x).view(zs.shape[0],-1,3)*1.2
		return x.view(zs.shape[0],-1,3)

# class LightDCTNet(nn.Module):
# 	def __init__(self,out_dim,dct_dim=None,weight_norm=True):
# 		super().__init__()
# 		if dct_dim is None:
# 			dims=[3,128,256,128,out_dim//5]
# 			self.register_buffer('dctbasis',utils.DCTSpace(out_dim//5,out_dim))
# 		else:
# 			dims=[3,128,256,128,dct_dim]
# 			self.register_buffer('dctbasis',utils.DCTSpace(dct_dim,out_dim))
# 		self.num_layers = len(dims)

# 		for l in range(0, self.num_layers - 1):
# 			out_dim = dims[l + 1]
# 			lin = nn.Linear(dims[l], out_dim)
# 			if weight_norm:
# 				lin = nn.utils.weight_norm(lin)

# 			setattr(self, "lin" + str(l), lin)

# 		self.relu = nn.ReLU()
# 		self.tanh = nn.Tanh()

# 	def forward(self, points):
# 		x = points

# 		for l in range(0, self.num_layers - 1):
# 			lin = getattr(self, "lin" + str(l))
# 			x = lin(x)
# 			if l < self.num_layers - 2:
# 				x = self.relu(x)
# 		x = self.tanh(x.matmul(self.dctbasis))
# 		return 2.*x

class LightDCTNet(nn.Module):
	def __init__(self,out_dim,dct_dim=None,weight_norm=True):
		super().__init__()
		if dct_dim is None:
			dims=[3,128,256,128,3*(out_dim//5)]
			self.register_buffer('dctbasis',utils.DCTSpace(out_dim//5,out_dim))
			self.dct_dim=out_dim//5
		else:
			dims=[3,128,256,128,3*dct_dim]
			self.register_buffer('dctbasis',utils.DCTSpace(dct_dim,out_dim))
			self.dct_dim=dct_dim
		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, points):
		x = points

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)
		# x = self.tanh(x.matmul(self.dctbasis))
		x = self.tanh(torch.cat([tx.matmul(self.dctbasis).unsqueeze(-1) for tx in torch.split(x,self.dct_dim,dim=-1)],dim=-1))
		return 2.*x

class LightDCTCondNet(nn.Module):
	def __init__(self,out_dim,feature_dim,dct_dim=None,weight_norm=True):
		super().__init__()
		if dct_dim is None:
			dims=[3+feature_dim,256,256,256,out_dim//5]
			self.register_buffer('dctbasis',utils.DCTSpace(out_dim//5,out_dim))
		else:
			dims=[3+feature_dim,256,256,256,dct_dim]
			self.register_buffer('dctbasis',utils.DCTSpace(dct_dim,out_dim))
		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

	def forward(self, points, zs):
		x = torch.cat([points,zs],dim=-1)

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)
		x = self.tanh(x.matmul(self.dctbasis))
		return 2.*x

class LightDCTCondNet2(nn.Module):
	def __init__(self,out_dim,feature_dim,dct_dim=None,weight_norm=True):
		super().__init__()
		if dct_dim is None:
			dims=[3+feature_dim,256,256,256,3*(out_dim//5)]
			self.register_buffer('dctbasis',utils.DCTSpace(out_dim//5,out_dim))
			self.dct_dim=out_dim//5
		else:
			dims=[3+feature_dim,256,256,256,3*dct_dim]
			self.register_buffer('dctbasis',utils.DCTSpace(dct_dim,out_dim))
			self.dct_dim=dct_dim
		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)
			if weight_norm:
				lin = nn.utils.weight_norm(lin)

			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()

	def forward(self, points, zs):
		x = torch.cat([points,zs],dim=-1)

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)
		return torch.cat([tx.matmul(self.dctbasis).unsqueeze(-1) for tx in torch.split(x,self.dct_dim,dim=-1)],dim=-1)

		

class LightGlobalNet(nn.Module):
	def __init__(self,out_dim,weight_norm=True):
		super().__init__()
		self.out_dim=out_dim
		self.register_parameter('light',nn.parameter.Parameter(torch.zeros(3,out_dim)))

	def forward(self, points):
		return self.light

class ShadingCorrectNet(nn.Module):
	def __init__(self,feature_dim,weight_norm=False):
		super().__init__()
		self.feature_dim=feature_dim
		dims=[3+feature_dim,128,256,128,3]
		self.num_layers = len(dims)

		for l in range(0, self.num_layers - 1):
			out_dim = dims[l + 1]
			lin = nn.Linear(dims[l], out_dim)
			if weight_norm:
				# lin = nn.utils.weight_norm(lin)
				print('ShadingCorrectNet:weight norm can influence weight initialization, can not produce small weights as initialization. Now do not use weight_norm')
			# initialize with zeros translation
			if l==self.num_layers-2:
				torch.nn.init.normal_(lin.weight, mean=0., std=0.001)
				torch.nn.init.constant_(lin.bias, 0.)
			setattr(self, "lin" + str(l), lin)

		self.relu = nn.ReLU()
	def forward(self,points,conds):
		x=torch.cat([points,conds],dim=-1)
		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))
			x = lin(x)
			if l < self.num_layers - 2:
				x = self.relu(x)
		return x



class RenderingNetwork_IDRSPH_framecond(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			frame_vector_size,
			mode,
			d_in,
			dims,
			weight_norm=True,
			multires_p=0,
			multires_n=0,
			out_dim=128,
			lightmodel='SPHLightNet'
	):
		super().__init__()
		self.featureTrans=FeatureTransformer(feature_vector_size,frame_vector_size,weight_norm)
		self.albedo=albedoNet(weight_norm,multires_p)
		self.light=globals()[lightmodel](out_dim,feature_vector_size,weight_norm)
		self.enable_px=True
		self.enable_framefeature=True	
		self.mode = mode
		dims = [d_in] + dims + [out_dim]


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

	def forward(self, points,xs, normals, view_dirs, feature_vectors,frame_vectors,ratio):
		albedos=self.albedo(points,1.)		
		ratio=ratio['renderRatio']
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))

		feature_vectors=self.featureTrans(feature_vectors,frame_vectors)
		lights=self.light(xs,feature_vectors)
		# rendering_input = torch.cat([normals, feature_vectors], dim=-1)
		# x = rendering_input
		x = normals

		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)

		x=x.unsqueeze(1).matmul(lights).squeeze(1)
		return (albedos+1.)*x-1.	#note albedo and render net color range(-1.,1.)


# class RenderingNetwork_albedo2(nn.Module):
# 	def __init__(
# 			self,
# 			feature_vector_size,
# 			weight_norm=True,
# 			multires_p=0,
# 			multires_n=0,
# 			out_dim=36,
# 			lightmodel='LightNet',
# 			dct_dim=None
# 	):
# 		super().__init__()
# 		self.albedo=albedoNet(weight_norm,multires_p)
# 		# self.light=LightNet(out_dim)
# 		self.lightmodel=lightmodel
# 		if 'Cond' in lightmodel:
# 			self.light=globals()[lightmodel](out_dim,feature_vector_size,dct_dim)
# 			dims=[3+3,256,512,256,3*out_dim]
# 		else:
# 			self.light=globals()[lightmodel](out_dim,dct_dim)
# 			dims=[3+3+feature_vector_size,256,512,256,3*out_dim]
# 		self.out_dim=out_dim
# 		self.embedn_fn = None
# 		self.multires_n=multires_n
# 		if multires_n > 0:
# 			embedn_fn, input_ch = get_embedder(multires_n)
# 			self.embedn_fn = embedn_fn
# 			dims[0] += (input_ch - 3)

# 		self.num_layers = len(dims)

# 		for l in range(0, self.num_layers - 1):
# 			out_dim = dims[l + 1]
# 			lin = nn.Linear(dims[l], out_dim)

# 			if weight_norm:
# 				lin = nn.utils.weight_norm(lin)

# 			setattr(self, "lin" + str(l), lin)

# 		self.relu = nn.ReLU()
# 		# self.sigmoid = nn.Sigmoid()


# 	def forward(self, points, normals, view_dirs, feature_vectors,ratio):
# 		ratio=ratio['renderRatio']		
# 		if self.embedn_fn is not None:
# 			if ratio is None:	#one weight, all equal one
# 				normals = self.embedn_fn(normals)
# 			elif ratio<=0: #zero weight
# 				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
# 			else:
# 				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))
# 		albedos=self.albedo(points,1.)
# 		if 'Cond' in self.lightmodel:
# 			lights=self.light(points,feature_vectors)
# 			x = torch.cat([view_dirs, normals], dim=-1)
# 		else:
# 			lights=self.light(points)
# 			x = torch.cat([view_dirs, normals, feature_vectors], dim=-1)


# 		for l in range(0, self.num_layers - 1):
# 			lin = getattr(self, "lin" + str(l))

# 			x = lin(x)

# 			if l < self.num_layers - 2:
# 				x = self.relu(x)
# 		x=(lights.unsqueeze(1).matmul(x.reshape(-1,self.out_dim,3))).reshape(-1,3)
# 		# x = self.sigmoid(x)
# 		return (albedos+1.)*x-1.	#note albedo and render net color range(-1.,1.)

class RenderingNetwork_albedo2(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			weight_norm=True,
			multires_p=0,
			multires_n=0,
			out_dim=36,
			lightmodel='LightNet',
			dct_dim=None
	):
		super().__init__()
		self.albedo=albedoNet(weight_norm,multires_p)
		# self.light=LightNet(out_dim)
		self.lightmodel=lightmodel
		if 'Cond' in lightmodel:
			self.light=globals()[lightmodel](out_dim,feature_vector_size,dct_dim)
			dims=[3+3,256,512,256,out_dim]
		else:
			self.light=globals()[lightmodel](out_dim,dct_dim)
			dims=[3+3+feature_vector_size,256,512,256,out_dim]
		self.out_dim=out_dim
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
		# self.sigmoid = nn.Sigmoid()


	def forward(self, points, normals, view_dirs, feature_vectors,ratio):
		ratio=ratio['renderRatio']		
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))
		albedos=self.albedo(points,1.)
		if 'Cond' in self.lightmodel:
			lights=self.light(points,feature_vectors)
			x = torch.cat([view_dirs, normals], dim=-1)
		else:
			lights=self.light(points)
			x = torch.cat([view_dirs, normals, feature_vectors], dim=-1)


		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)
		# x=(lights.matmul(x.reshape(-1,self.out_dim,1))).reshape(-1,3)
		x=x.unsqueeze(1).matmul(lights).squeeze(1)
		# x = self.sigmoid(x)
		return (albedos+1.)*x-1.	#note albedo and render net color range(-1.,1.)


class RenderingNetwork_albedo3(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			weight_norm=True,
			multires_p=0,
			out_dim=9
	):
		super().__init__()
		self.albedo=albedoNet(weight_norm,multires_p)
		# self.light=LightNet(out_dim)
		self.correct=ShadingCorrectNet(feature_vector_size)

		self.light=LightGlobalNet(out_dim)
		self.out_dim=out_dim
		if out_dim and out_dim!=9:
			dims=[3,256,512,256,out_dim]
			self.num_layers = len(dims)
			for l in range(0, self.num_layers - 1):
				out_dim = dims[l + 1]
				lin = nn.Linear(dims[l], out_dim)

				if weight_norm:
					lin = nn.utils.weight_norm(lin)

				setattr(self, "lin_" + str(l), lin)

			self.relu = nn.ReLU()

	def convert_normals(self,ns):
		return torch.stack([torch.ones(ns.shape[0],device=ns.device),ns[:,0],ns[:,1],ns[:,2], \
			ns[:,0]*ns[:,1],ns[:,0]*ns[:,2],ns[:,1]*ns[:,2],ns[:,0]*ns[:,0]-ns[:,1]*ns[:,1],3*ns[:,2]*ns[:,2]-1.])
	def forward(self, points, normals, view_dirs, feature_vectors,ratio):
		albedos=self.albedo(points,1.)
		if self.out_dim==9:
			shading=self.light(points).matmul(self.convert_normals(normals)).transpose(0,1)
		else:
			x=normals
			for l in range(0, self.num_layers - 1):
				lin = getattr(self, "lin_" + str(l))

				x = lin(x)

				if l < self.num_layers - 2:
					x = self.relu(x)
			shading=x.matmul(self.light(points).transpose(0,1))
		ds=self.correct(points,feature_vectors)
		
		x=shading+ds
		
		return (albedos+1.)*x-1.	#note albedo and render net color range(-1.,1.)



class RenderingNetwork_albedo4(nn.Module):
	def __init__(
			self,
			feature_vector_size,
			weight_norm=True,
			multires_p=0,
			multires_n=0,
			out_dim=36,
			lightmodel='LightDCTCondNet2',
			dct_dim=None
	):
		super().__init__()
		self.albedo=albedoNet(weight_norm,multires_p)
		# self.light=LightNet(out_dim)
		self.lightmodel=lightmodel
		assert(lightmodel=='LightDCTCondNet2')
		self.light=globals()[lightmodel](out_dim,feature_vector_size)
		dims=[3,256,512,256,out_dim]

		self.out_dim=out_dim
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
		# self.sigmoid = nn.Sigmoid()


	def forward(self, points, normals, view_dirs, feature_vectors,ratio):
		ratio=ratio['renderRatio']		
		if self.embedn_fn is not None:
			if ratio is None:	#one weight, all equal one
				normals = self.embedn_fn(normals)
			elif ratio<=0: #zero weight
				normals = self.embedn_fn(normals, [0. for _ in range(self.multires_n*2)])
			else:
				normals = self.embedn_fn(normals, utils.annealing_weights(self.multires_n,ratio))
		albedos=self.albedo(points,1.)
		
		lights=self.light(points,feature_vectors)
		x = normals


		for l in range(0, self.num_layers - 1):
			lin = getattr(self, "lin" + str(l))

			x = lin(x)

			if l < self.num_layers - 2:
				x = self.relu(x)
		x=x.unsqueeze(1).matmul(lights).squeeze(1)
		# x = self.sigmoid(x)
		return (albedos+1.)*x-1.	#note albedo and render net color range(-1.,1.)

# def getRenderNet(device,type=3):
# 	if type==1:
# 		net=RenderingNetwork(feature_vector_size=256,d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires=4)	
# 	elif type==2:
# 		net=RenderingNetwork_pos(feature_vector_size=256,d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires=4)
# 	elif type==3:
# 		net=RenderingNetwork_pos_norm(feature_vector_size=128,d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires=8)
# 	return net.to(device)

def getRenderNet(device,conf):
	if conf.get_string('type')=='RenderingNetwork_pos_norm2':
		return RenderingNetwork_pos_norm2(conf.get_int('condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires_p=conf.get_int('multires_p'),multires_n=conf.get_int('multires_n')).to(device)
	elif conf.get_string('type')=='RenderingNetwork_albedo':
		return RenderingNetwork_albedo(conf.get_int('condlen'),weight_norm=True,multires_p=conf.get_int('multires_p'),multires_n=conf.get_int('multires_n')).to(device)
	elif conf.get_string('type')=='RenderingNetwork_albedo2':
		return RenderingNetwork_albedo2(conf.get_int('condlen'), \
				dct_dim=conf.get_int('dct_dim') if 'dct_dim' in conf else None, \
				weight_norm=True, \
				multires_p=conf.get_int('multires_p'),multires_n=conf.get_int('multires_n'), \
				out_dim=conf.get_int('light_dim') if 'light_dim' in conf else 36, \
				lightmodel=conf.get_string('lightmodel') if 'lightmodel' in conf else 'LightNet').to(device)
	elif conf.get_string('type')=='RenderingNetwork_albedo3':
		return RenderingNetwork_albedo3(conf.get_int('condlen'), \
				weight_norm=True, \
				multires_p=conf.get_int('multires_p'), \
				out_dim=conf.get_int('light_dim') if 'light_dim' in conf else 9).to(device)
	elif conf.get_string('type')=='RenderingNetwork_albedo4':
		return RenderingNetwork_albedo4(conf.get_int('condlen'), \
				dct_dim=conf.get_int('dct_dim') if 'dct_dim' in conf else None, \
				weight_norm=True, \
				multires_p=conf.get_int('multires_p'),multires_n=conf.get_int('multires_n'), \
				out_dim=conf.get_int('light_dim') if 'light_dim' in conf else 36, \
				lightmodel=conf.get_string('lightmodel') if 'lightmodel' in conf else 'LightCondNet').to(device)
	elif conf.get_string('type')=='RenderingNetwork_view_norm':
		return RenderingNetwork_view_norm(conf.get_int('condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True, \
				multires_v=conf.get_int('multires_v'),multires_n=conf.get_int('multires_n')).to(device)
	elif conf.get_string('type')=='RenderingNetwork_IDRAug':
		return RenderingNetwork_IDRAug(conf.get_int('condlen'),d_in=12,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires_p=conf.get_int('multires_p'),multires_x=conf.get_int('multires_x'),multires_v=conf.get_int('multires_v'),multires_n=conf.get_int('multires_n')).to(device)
	elif conf.get_string('type')=='RenderingNetwork_IDRAug_framecond':
		return RenderingNetwork_IDRAug_framecond(conf.get_int('sdfcondlen'),conf.get_int('condlen'),d_in=12,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires_x=conf.get_int('multires_x'),multires_v=conf.get_int('multires_v'),multires_n=conf.get_int('multires_n')).to(device)
	elif conf.get_string('type')=='RenderingNetwork_IDRAlbedo':
		return RenderingNetwork_IDRAlbedo(conf.get_int('condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires_p=conf.get_int('multires_p'),multires_x=conf.get_int('multires_x'),multires_v=conf.get_int('multires_v'),multires_n=conf.get_int('multires_n')).to(device)
	elif conf.get_string('type')=='RenderingNetwork_IDRAlbedo_framecond':
		return RenderingNetwork_IDRAlbedo_framecond(conf.get_int('sdfcondlen'),conf.get_int('condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires_p=conf.get_int('multires_p'),multires_x=conf.get_int('multires_x'),multires_v=conf.get_int('multires_v'),multires_n=conf.get_int('multires_n')).to(device)
	elif conf.get_string('type')=='RenderingNetwork_IDRSPH_framecond':
		return RenderingNetwork_IDRSPH_framecond(conf.get_int('sdfcondlen'),conf.get_int('condlen'),d_in=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires_p=conf.get_int('multires_p'),multires_n=conf.get_int('multires_n'),out_dim=conf.get_int('out_dim'),lightmodel=conf.get_string('lightmodel')).to(device)
	else:
		return globals()[conf.get_string('type')](conf.get_int('condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires=conf.get_int('multires')).to(device)