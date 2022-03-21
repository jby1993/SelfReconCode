import torch
""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.create_embedding_fn()

	def create_embedding_fn(self):
		embed_fns = []
		d = self.kwargs['input_dims']
		out_dim = 0
		if self.kwargs['include_input']:
			embed_fns.append(lambda x: x)
			out_dim += d

		max_freq = self.kwargs['max_freq_log2']
		N_freqs = self.kwargs['num_freqs']

		if self.kwargs['log_sampling']:
			freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
		else:
			freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

		for freq in freq_bands:
			for p_fn in self.kwargs['periodic_fns']:
				embed_fns.append(lambda x, weight=1., p_fn=p_fn,
								 freq=freq: weight*p_fn(x * freq))
				out_dim += d

		self.embed_fns = embed_fns
		self.out_dim = out_dim

	def embed(self, inputs, ws=None):
		if ws is None:
			return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
		else:
			if self.kwargs['include_input']:
				return torch.cat([self.embed_fns[0](inputs)]+[fn(inputs,w) for fn,w in zip(self.embed_fns[1:],ws)], -1)
			else:
				return torch.cat([fn(inputs,w) for fn,w in zip(self.embed_fns,ws)], -1)

def get_embedder(multires):
	embed_kwargs = {
		'include_input': True,
		'input_dims': 3,
		'max_freq_log2': multires-1,
		'num_freqs': multires,
		'log_sampling': True,
		'periodic_fns': [torch.sin, torch.cos],
	}

	embedder_obj = Embedder(**embed_kwargs)
	def embed(x, ws=None, eo=embedder_obj): return eo.embed(x,ws)
	return embed, embedder_obj.out_dim