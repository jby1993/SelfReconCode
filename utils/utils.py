import numpy as np
import torch
from torch_scatter import scatter
from FastMinv import Fast3x3Minv,Fast3x3Minv_backward

from torch.autograd import Function

class FastDiff3x3MinvFunction(Function):
	@staticmethod
	def forward(ctx,input):		
		invs,check=Fast3x3Minv(input)
		ctx.save_for_backward(invs,check)
		ctx.mark_non_differentiable(check)
		return invs,check
	@staticmethod
	def backward(ctx, grad_input, grad_check):
		invs, check = ctx.saved_tensors
		return Fast3x3Minv_backward(grad_input.contiguous(),invs),None


def quat2mat(quat):
	"""Convert quaternion coefficients to rotation matrix.
	Args:
		quat: size = [B, 4] 4 <===>(w, x, y, z)
	Returns:
		Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
	"""
	norm_quat = quat
	norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
	w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]
	B = quat.size(0)
	w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
	wx, wy, wz = w*x, w*y, w*z
	xy, xz, yz = x*y, x*z, y*z
	rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
						  2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
						  2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
	return rotMat

def annealing_weights(multires,ratio):
	alpha=ratio*multires
	out=[]
	for ind in range(multires):
		w=(1.-np.cos(np.pi*min(max(alpha-float(ind),0.),1.)))/2.
		out.extend([w,w])
	return out

def GMRobustError(x,c,square=False):
	if square:
		return 2.*x/(c*c)/(x/(c*c)+4)
	else:
		return 2.*x*x/(c*c)/(x*x/(c*c)+4)



def smpl_tmp_Apose(init_pose_type=0):
	pose=np.zeros((24,3))
	if init_pose_type==0:		
		pose[1]=np.array([0,0,10./180.*np.pi])
		pose[2]=np.array([0,0,-10./180.*np.pi])

		pose[16]=np.array([0,0,-45./180.*np.pi])
		pose[17]=np.array([0,0,45./180.*np.pi])
	elif init_pose_type==1:
		pose[1]=np.array([0,0,7./180.*np.pi])
		pose[2]=np.array([0,0,-7./180.*np.pi])

		pose[16]=np.array([0,0,-55./180.*np.pi])
		pose[17]=np.array([0,0,55./180.*np.pi])
	else:
		assert(False)
	return pose.astype(np.float32)

def sample_points(pc_input, global_sigma, local_sigma, ratio=6):
	sample_size, dim = pc_input.shape

	sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)
	if ratio>0:
		sample_global = (torch.rand(sample_size // ratio, dim, device=pc_input.device) * (global_sigma * 2)) - global_sigma
		sample = torch.cat([sample_local, sample_global], dim=0)
	else:
		sample = sample_local

	return sample

def compute_Jacobian_debug(ps,ds,retain_graph,create_graph,allow_unused=False):
	grad_d_p=[]
	grad_outputs=torch.zeros_like(ds)
	grad_outputs[...,0]=1.
	outx=torch.autograd.grad(ds,ps,grad_outputs,retain_graph=True,create_graph=create_graph,allow_unused=allow_unused)
	grad_d_p.append(outx[0].view(-1,1,3))
	
	grad_outputs.zero_()
	grad_outputs[...,1]=1.
	outy=torch.autograd.grad(ds,ps,grad_outputs,retain_graph=True,create_graph=create_graph,allow_unused=allow_unused)
	grad_d_p.append(outy[0].view(-1,1,3))

	grad_outputs.zero_()
	grad_outputs[...,2]=1.
	outz=torch.autograd.grad(ds,ps,grad_outputs,retain_graph=retain_graph,create_graph=create_graph,allow_unused=allow_unused)
	grad_d_p.append(outz[0].view(-1,1,3))	
	
	grad_d_p=torch.cat(grad_d_p,dim=1) #N,3,3
	return grad_d_p

def compute_Jacobian(ps,ds,retain_graph,create_graph,allow_unused=False):
	grad_d_p=[]
	grad_outputs=torch.ones_like(ds[...,0])
	outx=torch.autograd.grad(ds[...,0],ps,grad_outputs,retain_graph=True,create_graph=create_graph,allow_unused=allow_unused)
	grad_d_p.append(outx[0].view(-1,1,3))
	
	outy=torch.autograd.grad(ds[...,1],ps,grad_outputs,retain_graph=True,create_graph=create_graph,allow_unused=allow_unused)
	grad_d_p.append(outy[0].view(-1,1,3))
	

	outz=torch.autograd.grad(ds[...,2],ps,grad_outputs,retain_graph=retain_graph,create_graph=create_graph,allow_unused=allow_unused)
	grad_d_p.append(outz[0].view(-1,1,3))
	
	grad_d_p=torch.cat(grad_d_p,dim=1) #N,3,3
	return grad_d_p

# def compute_deformed_normals(sdf,deformer,ps,defconds,batch_inds,ratio,phase):
# 	sdfs=sdf(ps,ratio)
# 	check=True if phase=='train' or phase=='Train' else False
# 	nx=torch.autograd.grad(sdfs,ps,torch.ones_like(sdfs),retain_graph=check,create_graph=check)[0]
# 	ds=deformer(ps,defconds,batch_inds,ratio=ratio)
# 	grad_d_p=compute_Jacobian(ps,ds,check,check)
# 	nx=grad_d_p.matmul(nx.view(-1,3,1)).view(-1,3)
# 	nx=nx/nx.norm(dim=1,keepdim=True)
# 	return nx

def compute_deformed_normals(sdf,deformer,ps,defconds,batch_inds,ratio,phase):
	sdfs=sdf(ps,ratio)
	check=True if phase=='train' or phase=='Train' else False
	onx=torch.autograd.grad(sdfs,ps,torch.ones_like(sdfs),retain_graph=check,create_graph=check)[0]
	if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
		if not check:
			sdfs=sdf(ps,ratio)
		ds=deformer(ps,[[defconds[0],sdf.rendcond],defconds[1]],batch_inds,ratio=ratio)
	else:
		ds=deformer(ps,defconds,batch_inds,ratio=ratio)
	grad_d_p=compute_Jacobian(ps,ds,check,check)
	grad_d_p_inv,inv_mask=FastDiff3x3MinvFunction.apply(grad_d_p)
	nx=grad_d_p_inv.transpose(-2,-1).matmul(onx.view(-1,3,1)).view(-1,3)
	n_inv_mask=~inv_mask
	if n_inv_mask.sum().item()>0:
		print('unwished error n_inv_mask:(%d:%d)'%(n_inv_mask.sum().item(),n_inv_mask.numel()))
		nnx=torch.zeros_like(nx)
		nnx[inv_mask]=nx[inv_mask]
		nnx[n_inv_mask]=grad_d_p[n_inv_mask].matmul(onx[n_inv_mask].unsqueeze(-1)).view(-1,3)
		nx=nnx
	nx=nx/nx.norm(dim=1,keepdim=True)
	return nx,ds

def compute_cardinal_rays(deformer,ps,rays,defconds,batch_inds,ratio,phase):
	check=True if phase=='train' or phase=='Train' else False
	ds=deformer(ps,defconds,batch_inds,ratio=ratio)
	grad_d_p=compute_Jacobian(ps,ds,check,check)
	grad_d_p_inv,inv_mask=FastDiff3x3MinvFunction.apply(grad_d_p)
	crays=grad_d_p_inv.matmul(rays.view(-1,3,1)).view(-1,3)
	n_inv_mask=~inv_mask
	if n_inv_mask.sum().item()>0:
		print('unwished error n_inv_mask:(%d:%d)'%(n_inv_mask.sum().item(),n_inv_mask.numel()))
		ncrays=torch.zeros_like(crays)
		ncrays[inv_mask]=crays[inv_mask]
		ncrays[n_inv_mask]=rays[n_inv_mask].detach()
		crays=ncrays
	crays=crays/crays.norm(dim=1,keepdim=True)
	return crays,ds

def compute_netRender_color(net,ps,ds,ns,vs,features,framefeatures,ratio,idr_like_cond,idr_xs_render):
	if hasattr(net,'enable_px'):
		if hasattr(net,'enable_framefeature'):
			return net(ps, ds, ns, vs, features,framefeatures,ratio)
		else:
			return net(ps, ds, ns, vs, features if idr_like_cond else framefeatures,ratio)
	else:
		if hasattr(net,'enable_framefeature'):
			return net(ds if idr_xs_render else ps, ns, vs, features, framefeatures,ratio)
		else:
			return net(ds if idr_xs_render else ps, ns, vs, features if idr_like_cond else framefeatures,ratio)

# verts(N,vnum,3), faces(fnum,3) or (N,fnum,3)
def compute_face_areas(verts,faces):
	N,vnum,_ = verts.shape
	if faces.dim()==2:
		faces=faces.view(1,-1,3).expand(N,-1,3)
	assert(faces.shape[0]==N)
	assert(verts.shape[-1]==faces.shape[-1])
	fnum=faces.shape[1]

	fvs=torch.gather(verts,1,faces.reshape(N,-1,1).repeat(1,1,3)).reshape(N,fnum,3,3)
	v01=fvs[:,:,1,:]-fvs[:,:,0,:]
	v02=fvs[:,:,2,:]-fvs[:,:,0,:]
	return torch.cross(v01,v02,dim=-1).norm(dim=-1)/2.

#verts:(v,3) or (b,v,3), tri_fs:(f,3)
def compute_fnorms(verts,tri_fs):
	v0=verts.index_select(-2,tri_fs[:,0])
	v1=verts.index_select(-2,tri_fs[:,1])
	v2=verts.index_select(-2,tri_fs[:,2])
	e01=v1-v0
	e02=v2-v0
	fnorms=torch.cross(e01,e02,-1)
	diss=fnorms.norm(2,-1).unsqueeze(-1)
	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
	fnorms=fnorms/diss
	return fnorms

def DCTBasis(k,N):
	assert(k<N)
	basis=torch.tensor([np.pi*(float(n)+0.5)*k/float(N) for n in range(N)]).float()
	basis=torch.cos(basis)*(1./np.sqrt(float(N)) if k==0 else np.sqrt(2./float(N)))
	return basis

def DCTNullSpace(k,N):
	return torch.stack([DCTBasis(ind,N) for ind in range(k,N)])

def DCTSpace(k,N):
	return torch.stack([DCTBasis(ind,N) for ind in range(0,k)])


# def compute_vnorms(verts,tri_fs,vertex_index,face_index):
# 	fnorms=compute_fnorms(verts,tri_fs)
# 	vnorms=geo_utils.scatter_('add',fnorms[face_index,:],vertex_index)
# 	diss=vnorms.norm(2,1).unsqueeze(-1)
# 	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
# 	vnorms=vnorms/diss
# 	return vnorms
#verts(b,vnum,3) or (vnum,3)
def compute_vnorms(verts,tri_fs,vertex_index,face_index):
	fnorms=compute_fnorms(verts,tri_fs)
	vnorms=scatter(fnorms.index_select(-2,face_index),vertex_index,-2,dim_size=verts.shape[-2],reduce='sum')
	diss=vnorms.norm(2,-1).unsqueeze(-1)
	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
	vnorms=vnorms/diss
	return vnorms

from MCAcc import Seg3dLossless
from pytorch3d.renderer import (
    RasterizationSettings, 
    BlendParams
)
from model.CameraMine import RectifiedPerspectiveCameras

def set_hierarchical_config(conf,name,optNet,dataloader,resolutions):
	if 'point_render' in conf.get_config('train.'+name):
		batch_size=conf.get_int('train.'+name+'.point_render.batch_size')
	else:
		batch_size=conf.get_int('train.'+name+'.batch_size')
	dataloader=torch.utils.data.DataLoader(dataloader.dataset,batch_size,sampler=dataloader.sampler,num_workers=dataloader.num_workers)
	# sigma=conf.get_float('train.'+name+'.sigma')	
	# optNet.next_remesh_intersect=conf.get_int('train.'+name+'.remesh_intersect')
	# optNet.next_conf=conf.get_config('loss_'+name)
	# optNet.next_raset = RasterizationSettings(
	# 	image_size=(optNet.maskRender.rasterizer.raster_settings.image_size[0],optNet.maskRender.rasterizer.raster_settings.image_size[1]), 
	# 	blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
	# 	# bin_size=0,
	# 	faces_per_pixel=conf.get_int('train.'+name+'.faces_per_pixel'),
	# 	perspective_correct=True,
	# 	clip_barycentric_coords=False,
	# 	cull_backfaces=optNet.maskRender.rasterizer.raster_settings.cull_backfaces
	# )
	# optNet.next_shaderset = BlendParams(sigma=sigma)
	optNet.next_conf=conf.get_config('loss_'+name)
	optNet.next_train_conf=conf.get_config('train.'+name)
	engine = Seg3dLossless(
			query_func=None, 
			b_min = optNet.engine.b_min,
			b_max = optNet.engine.b_max,
			resolutions=resolutions,
			align_corners=False,
			balance_value=0.0, # be careful 
			visualize=False,
			debug=False,
			use_cuda_impl=False,
			faster=False 
		).to(optNet.engine.b_min.device)
	optNet.engine=engine
	return optNet,dataloader

def save_model(name,epoch,optNet,dataset):
	outdic={"epoch": epoch, "model_state_dict": optNet.state_dict()}
	outdic.update(dataset.camera_params)
	# if idr_like_cond:
	# 	outdic.update({'poses':dataset.poses,'trans':dataset.trans,'shape':dataset.shape,'dcond':dataset.conds[0]})
	# else:
	outdic.update({'poses':dataset.poses,'trans':dataset.trans,'shape':dataset.shape,'dcond':dataset.conds[0],'rcond':dataset.conds[1]})
	torch.save(outdic, name)

def load_model(name,optNet,dataset,device,subsdfmodel=None,model_rm_prefix=None):
	saved_model_state = torch.load(name,map_location='cpu')
	optNet_model_state={k:v for k,v in saved_model_state["model_state_dict"].items() if 'engine.' not in k}	
	if model_rm_prefix:
		tmp={}
		for k,v in optNet_model_state.items():
			rm_ok=False
			for prefix in model_rm_prefix:
				if len(k)>=len(prefix) and k[:len(prefix)]==prefix:
					rm_ok=True
			if not rm_ok:
				tmp[k]=v
		optNet_model_state=tmp
	if subsdfmodel is not None:
		sdf_model=torch.load(subsdfmodel,map_location='cpu')
		optNet_model_state={k:v for k,v in optNet_model_state.items() if 'sdf.' not in k}
		optNet_model_state.update({'sdf.'+k:v for k,v in sdf_model.items()})

	# before ws computation has trunctation bug(pleated skinner deformation), so delete the weights and keep load initial_skinner_ws weights
	optNet_model_state={k:v for k,v in optNet_model_state.items() if 'deformer.defs.1.ws' not in k}


	optNet.load_state_dict(optNet_model_state,strict=False)
	optNet=optNet.to(device)
	# if not idr_like_cond:
	# dataset.conds=[saved_model_state['dcond'].requires_grad_(),saved_model_state['rcond'].requires_grad_()]
	dataset.conds[0] = saved_model_state['dcond'].requires_grad_() if 'dcond' in saved_model_state else dataset.conds[0]
	dataset.conds[1] = saved_model_state['rcond'].requires_grad_() if 'rcond' in saved_model_state else dataset.conds[1]
	# else:
	# 	dataset.conds=[saved_model_state['dcond'].requires_grad_()]
	grad=dataset.poses.requires_grad
	dataset.poses=saved_model_state['poses'].requires_grad_(grad)
	assert(dataset.frame_num<=dataset.poses.shape[0])
	grad=dataset.trans.requires_grad
	dataset.trans=saved_model_state['trans'].requires_grad_(grad)
	assert(dataset.frame_num<=dataset.trans.shape[0])
	grad=dataset.shape.requires_grad
	dataset.shape=saved_model_state['shape'].requires_grad_(grad)
	camera_params={}
	for k,v in dataset.camera_params.items():
		camera_params[k]=saved_model_state[k].requires_grad_(v.requires_grad)
	dataset.camera_params=camera_params

	cam_data=dataset.camera_params
	N=optNet.maskRender.rasterizer.cameras._N
	cameras=RectifiedPerspectiveCameras(cam_data['focal_length'].view(1,2).expand(N,2),cam_data['princeple_points'].view(1,2).expand(N,2),
				quat2mat(cam_data['cam2world_coord_quat'].view(1,4)).expand(N,3,3),cam_data['world2cam_coord_trans'].view(1,3).expand(N,3),
				image_size=[(dataset.W, dataset.H)]).to(device)
	optNet.maskRender.rasterizer.cameras=cameras

	return optNet,dataset




def save_rigid_model(name,epoch,optNet,dataset,idr_like_cond):
	outdic={"epoch": epoch, "model_state_dict": optNet.state_dict()}
	outdic.update(dataset.camera_params)
	if idr_like_cond:
		outdic.update({'Rs':dataset.Rs,'trans':dataset.trans})
	else:
		outdic.update({'Rs':dataset.Rs,'trans':dataset.trans,'rcond':dataset.conds[0]})
	torch.save(outdic, name)

def load_rigid_model(name,optNet,dataset,device,idr_like_cond,subsdfmodel=None):
	saved_model_state = torch.load(name,map_location='cpu')
	optNet_model_state={k:v for k,v in saved_model_state["model_state_dict"].items() if 'engine.' not in k}	
	if subsdfmodel is not None:
		sdf_model=torch.load(subsdfmodel,map_location='cpu')
		optNet_model_state={k:v for k,v in optNet_model_state.items() if 'sdf.' not in k}
		optNet_model_state.update({'sdf.'+k:v for k,v in sdf_model.items()})

	optNet.load_state_dict(optNet_model_state,strict=False)
	optNet=optNet.to(device)
	if not idr_like_cond:
		dataset.conds=[saved_model_state['rcond'].requires_grad_()]
	grad=dataset.Rs.requires_grad
	dataset.Rs=saved_model_state['Rs'].requires_grad_(grad)
	assert(dataset.frame_num<=dataset.Rs.shape[0])
	grad=dataset.trans.requires_grad
	dataset.trans=saved_model_state['trans'].requires_grad_(grad)
	assert(dataset.frame_num<=dataset.trans.shape[0])
	camera_params={}
	for k,v in dataset.camera_params.items():
		camera_params[k]=saved_model_state[k].requires_grad_(v.requires_grad)
	dataset.camera_params=camera_params

	cam_data=dataset.camera_params
	N=optNet.maskRender.rasterizer.cameras._N
	cameras=RectifiedPerspectiveCameras(cam_data['focal_length'].view(1,2).expand(N,2),cam_data['princeple_points'].view(1,2).expand(N,2),
				quat2mat(cam_data['cam2world_coord_quat'].view(1,4)).expand(N,3,3),cam_data['world2cam_coord_trans'].view(1,3).expand(N,3),
				image_size=[(dataset.W, dataset.H)]).to(device)
	optNet.maskRender.rasterizer.cameras=cameras

	return optNet,dataset