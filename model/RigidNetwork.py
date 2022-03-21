import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.autograd.functional as F
from FastMinv import Fast3x3Minv
from .Embedder import get_embedder
import utils
import os
import os.path as osp

import MCGpu
from MCAcc import Seg3dLossless
from pytorch3d.structures import Meshes,Pointclouds,join_meshes_as_batch
from pytorch3d.loss import (
	chamfer_distance, 
	mesh_edge_loss, 
	mesh_laplacian_smoothing, 
	mesh_normal_consistency,
)
from .CameraMine import PointsRendererWithFrags,RectifiedPerspectiveCameras
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.io import load_objs_as_meshes
from torch_scatter import scatter
import model.RenderNet as RenderNet
from pytorch3d.renderer import (
	RasterizationSettings, 
	MeshRasterizer,
	SoftSilhouetteShader,
	TexturesVertex,
	BlendParams,
	PointsRasterizationSettings,
	# PointsRenderer,
	PointsRasterizer,
	PointLights,
	AlphaCompositor
)
import openmesh as om
import trimesh
import cv2
class OptimRigidNetwork(nn.Module):
	def __init__(self,TmpSdf,Deformer,accEngine,maskRender,netRender,conf=None):
		super().__init__()
		self.conf=conf
		self.sdf=TmpSdf
		self.deformer=Deformer
		self.maskRender=maskRender
		self.netRender=netRender
		self.engine=accEngine
		self.angThred=self.maskRender.rasterizer.cameras.angThreshold(0.5)
		print('camera ang threshold is %f'%self.angThred)
		self.TmpVs=None
		self.Tmpfs=None
		self.forward_time=0
		self.remesh_intersect=30
		self.remesh_time=0.
		self.next_conf=None
		self.next_train_conf=None
		self.pcRender=None
		self.draw=False
		self.enable_mesh_color=True
		self.sdfShrinkRadius=0.0
		self.only_surface_ps=False
		self.idr_like_render=False
		self.idr_like_cond=False
	def update_hierarchical_config(self,device):
		if self.next_conf is not None:
			self.conf=self.next_conf
			self.forward_time=0
			rasterizer=self.maskRender.rasterizer
			if 'point_render' in self.next_train_conf:
				raster_settings_silhouette = PointsRasterizationSettings(
					image_size=(rasterizer.raster_settings.image_size[0],rasterizer.raster_settings.image_size[1]), 
					radius=self.next_train_conf.get_float('point_render.radius'),
					bin_size=64,
					points_per_pixel=50,
					)   
				self.pcRender=PointsRendererWithFrags(
					rasterizer=PointsRasterizer(
						cameras=rasterizer.cameras, 
						raster_settings=raster_settings_silhouette
					),
						compositor=AlphaCompositor(background_color=None)
					).to(device)
				self.remesh_intersect=self.next_train_conf.get_int('point_render.remesh_intersect')
				self.sdfShrinkRadius=0.0
				raster_settings_silhouette=RasterizationSettings(
						image_size=(rasterizer.raster_settings.image_size[0],rasterizer.raster_settings.image_size[1]), 
						blur_radius=0.,
						# blur_radius=np.log(1. / 1e-4 - 1.)*3.e-6,
						faces_per_pixel=1,
						perspective_correct=True,
						clip_barycentric_coords=False,
						cull_backfaces=self.maskRender.rasterizer.raster_settings.cull_backfaces
						# cull_backfaces=False
					)
				self.maskRender.rasterizer.raster_settings=raster_settings_silhouette
			else:
				self.pcRender=None
				sigma=self.next_train_conf.get_float('sigma')	
				self.remesh_intersect=self.next_train_conf.get_int('remesh_intersect')
				raster_settings_silhouette=RasterizationSettings(
						image_size=(rasterizer.raster_settings.image_size[0],rasterizer.raster_settings.image_size[1]), 
						blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
						# bin_size=0,
						faces_per_pixel=self.next_train_conf.get_int('faces_per_pixel'),
						perspective_correct=True,
						clip_barycentric_coords=False,
						cull_backfaces=self.maskRender.rasterizer.raster_settings.cull_backfaces
					)				
				self.maskRender.rasterizer.raster_settings=raster_settings_silhouette
				self.maskRender.shader.blend_params = BlendParams(sigma=sigma)			
			self.next_conf=None
			self.next_train_conf=None

	def discretizeSDF(self,ratio,engine=None,balance_value=0.):
		def query_func(points):
			with torch.no_grad():
				return self.sdf.forward(points.reshape(-1,3),ratio).reshape(1,1,-1)
		if engine is None:
			engine=self.engine
		engine.balance_value=balance_value
		engine.query_func=query_func
		sdfs=engine.forward()		
		verts, faces=MCGpu.mc_gpu(sdfs[0,0].permute(2,1,0).contiguous(),engine.spacing_x,engine.spacing_y,engine.spacing_z,engine.bx,engine.by,engine.bz,balance_value)
		return verts,faces
	#gtCs(N,H,W,3): colors
	#gtMs(N,H,W): masks

	def infer(self,TmpVs,Tmpfs,H,W,ratio,frame_ids,notcolor=False,gts=None):
		device=TmpVs.device
		with torch.no_grad():
			focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
			cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
			self.maskRender.rasterizer.cameras=cameras
			if self.pcRender:
				self.pcRender.rasterizer.cameras=cameras

			TmpVnum=TmpVs.shape[0]
			N=frame_ids.numel()
			Rs,trans,rendcond=self.dataset.get_grad_parameters(frame_ids,device)
			defTmpVs=self.deformer(TmpVs[None,:,:].expand(N,-1,3),[Rs,trans],ratio=ratio)
			defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))

			for ind,(vs,fs) in enumerate(zip(defMeshes.verts_list(),defMeshes.faces_list())):
				mesh = trimesh.Trimesh(vs.view(TmpVs.shape[0],3).detach().cpu().numpy(), fs.cpu().numpy())
				mesh.export('def_%d.ply'%ind)

			imgs,frags=self.maskRender(defMeshes)			
			if gts:
				masks=(frags.pix_to_face>=0).float()[...,0]
				gtMs=gts['mask']
				gts['maskE']=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).cpu().numpy()
				masks=masks>0.
				imgs=imgs[...,:3]
				if 'image' in gts:
					imgs[~masks]=gts['image'][~masks][:,[2,1,0]]


			imgs=torch.clamp(imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)

			if self.pcRender is not None and False:
				if hasattr(self.netRender,'albedo'):
					temp=self.netRender.albedo(TmpVs,1.).view(-1,3)
					features=[torch.cat([temp,torch.ones(TmpVs.shape[0],1,device=TmpVs.device)],dim=-1)]*N
					# features=[torch.ones(self.TmpVs.shape[0],1,device=self.TmpVs.device)]*N
				else:
					features=[torch.ones(TmpVs.shape[0],1,device=TmpVs.device) for _ in range(N)]
				pcmasks=self.pcRender(Pointclouds(points=defMeshes.verts_list(),features=features))
				pcmasks[frags.pix_to_face[...,0]<0]=torch.ones(1,4,device=device)
				if pcmasks.shape[-1]>=4:
					pcmasks=torch.clamp((pcmasks[...,0:3]+1.)*255./2.,min=0.,max=255.)
					pcmasks=pcmasks.detach().cpu().numpy().astype(np.uint8)
				else:
					pcmasks=(pcmasks*255.).detach().cpu().numpy().astype(np.uint8)
			else:
				pcmasks=None

			batch_inds,row_inds,col_inds,initTmpPs,_=utils.FindSurfacePs(TmpVs.detach(),Tmpfs,frags)

			cameras=self.maskRender.rasterizer.cameras
			rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())
			defconds=[Rs.detach(),trans.detach()]
		if notcolor:
			return None,None,imgs,def1imgs,pcmasks
		tcolors=[]
		talbedos=[]
		# nxs=[]
		# nx2s=[]
		print('draw %d points'%rays.shape[0])
		for ind,(rays_,initTmpPs_,batch_inds_) in enumerate(zip(torch.split(rays,10000),torch.split(initTmpPs,10000),torch.split(batch_inds,10000))):
			initTmpPs_,check=utils.OptimizeSurfacePs(cameras.cam_pos().detach(),rays_.detach(),initTmpPs_.clone(),batch_inds_,self.sdf,ratio,self.deformer,defconds,dthreshold=5.e-5,athreshold=self.angThred,w1=3.05,w2=1.,times=30)
			# print('%d:(%d,%d)'%(ind,rays_.shape[0],check.sum().item()))
			initTmpPs_.requires_grad=True
			if not self.idr_like_render:
				nx=utils.compute_deformed_normals(self.sdf,self.deformer,initTmpPs_,defconds,batch_inds_,ratio,'test')[0]
			else:
				sdfs=self.sdf(initTmpPs_,ratio)
				nx=torch.autograd.grad(sdfs,initTmpPs_,torch.ones_like(sdfs),retain_graph=False,create_graph=False)[0]
				nx=nx/nx.norm(dim=1,keepdim=True)
				rays_=utils.compute_cardinal_rays(self.deformer,initTmpPs_,rays_,defconds,batch_inds_,ratio,'test')[0]
			with torch.no_grad():
				tcolors.append(self.netRender(initTmpPs_, nx, -rays_, self.sdf.rendcond if self.idr_like_cond else rendcond[batch_inds_],ratio))
				if hasattr(self.netRender,'albedo'):
					talbedos.append(self.netRender.albedo(initTmpPs_,1.))

		tcolors=torch.cat(tcolors,dim=0)
		tcolors=torch.clamp((tcolors/2.+0.5)*255.,min=0.,max=255.)
		colors=torch.ones(N,H,W,3,device=device)*255.
		colors[batch_inds,row_inds,col_inds,:]=tcolors
		if gts and 'image' in gts:
			colors[~masks]=gts['image'][~masks][:,:3]*255.

		colors=colors.cpu().numpy().astype(np.uint8)
		if len(talbedos):
			talbedos=torch.cat(talbedos,dim=0)
			talbedos=torch.clamp((talbedos/2.+0.5)*255.,min=0.,max=255.)
			albedos=torch.ones(N,H,W,3,device=device)*255.
			albedos[batch_inds,row_inds,col_inds,:]=talbedos
			albedos=albedos.cpu().numpy().astype(np.uint8)
		else:
			albedos=None
		
		return colors,albedos,imgs,def1imgs,pcmasks

	def save_debug(self,TmpVs,Tmpfs,defMeshes,masks,gtMs,mgtMs,gtAs,gtCs,batch_inds,row_inds,col_inds,initTmpPs,defconds,rendcond,ratio):
		# mesh = trimesh.Trimesh(TmpVs.detach().cpu().numpy(), Tmpfs.cpu().numpy())
		# mesh.export(osp.join('tmp.ply'))
		if self.root is None:
			return

		mesh = trimesh.Trimesh(TmpVs.detach().cpu().numpy(), Tmpfs.cpu().numpy())
		mesh.export(osp.join(self.root,'tmp.ply'))	

		for ind,(vs,fs) in enumerate(zip(defMeshes.verts_list(),defMeshes.faces_list())):
			mesh = trimesh.Trimesh(vs.view(TmpVs.shape[0],3).detach().cpu().numpy(), fs.cpu().numpy())
			mesh.export(osp.join(self.root,'def_%d.ply'%ind))
		N=gtMs.shape[0]

		if masks.shape[-1]>=4:
			images=torch.clamp((masks[...,0:3]+1.)*255./2.,min=0.,max=255.)
			images=images.detach().cpu().numpy().astype(np.uint8)
			gtMasks=((gtMs.unsqueeze(-1)*gtAs+1.)*255./2.).detach().cpu().numpy().astype(np.uint8)
		else:
			images=(masks*255.).detach().cpu().numpy().astype(np.uint8)
			gtMasks=(gtMs*255.).detach().cpu().numpy().astype(np.uint8)			
		for ind,(img,gtimg) in enumerate(zip(images,gtMasks)):
			cv2.imwrite(osp.join(self.root,'m%d.png'%ind),img)
			cv2.imwrite(osp.join(self.root,'gm%d.png'%ind),gtimg)
		if mgtMs is not None:
			mgtMasks=(mgtMs*255.).detach().cpu().numpy().astype(np.uint8)	
			for ind,mgtimg in enumerate(mgtMasks):
				cv2.imwrite(osp.join(self.root,'mgm%d.png'%ind),mgtimg)

		#debug draw the images
		if self.draw:			
			cameras=self.maskRender.rasterizer.cameras
			with torch.no_grad():
				rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())
			# defconds=[d_cond.detach(),[poses.detach(),trans.detach()]]
			tcolors=[]
			talbedos=[]
			tcorrects=[]
			# tlights=[]
			print('draw %d points'%rays.shape[0])
			for ind,(rays_,initTmpPs_,batch_inds_) in enumerate(zip(torch.split(rays,20000),torch.split(initTmpPs,20000),torch.split(batch_inds,20000))):
				initTmpPs_,check=utils.OptimizeSurfacePs(cameras.cam_pos().detach(),rays_.detach(),initTmpPs_.clone(),batch_inds_,self.sdf,ratio,self.deformer,defconds,dthreshold=5.e-5,athreshold=self.angThred,w1=3.05,w2=1.,times=30)
				# print('%d:(%d,%d)'%(ind,rays_.shape[0],check.sum().item()))
				initTmpPs_.requires_grad=True
				if not self.idr_like_render:
					nx=utils.compute_deformed_normals(self.sdf,self.deformer,initTmpPs_,defconds,batch_inds_,ratio,'test')[0]
				else:
					sdfs=self.sdf(initTmpPs_,ratio)
					nx=torch.autograd.grad(sdfs,initTmpPs_,torch.ones_like(sdfs),retain_graph=False,create_graph=False)[0]
					nx=nx/nx.norm(dim=1,keepdim=True)
					rays_=utils.compute_cardinal_rays(self.deformer,initTmpPs_,rays_,defconds,batch_inds_,ratio,'test')[0]
				with torch.no_grad():
					tcolors.append(self.netRender(initTmpPs_, nx, -rays_, self.sdf.rendcond if self.idr_like_cond else rendcond[batch_inds_],ratio))
					if hasattr(self.netRender,'albedo'):
						talbedos.append(self.netRender.albedo(initTmpPs_,1.))
					if hasattr(self.netRender,'correct'):
						tmp=self.netRender.correct(initTmpPs_,rendcond[batch_inds_])
						tcorrects.append(tcolors[-1]-(talbedos[-1]+1.)*tmp)
			tcolors=torch.cat(tcolors,dim=0)			
			# print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
			tcolors=torch.clamp((tcolors/2.+0.5)*255.,min=0.,max=255.)
			colors=torch.ones_like(gtCs)*255.
			colors[batch_inds,row_inds,col_inds,:]=tcolors
			colors=colors.cpu().numpy().astype(np.uint8)
			if len(talbedos):
				talbedos=torch.cat(talbedos,dim=0)
				# print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
				talbedos=torch.clamp((talbedos/2.+0.5)*255.,min=0.,max=255.)
				albedos=torch.ones_like(gtCs)*255.
				albedos[batch_inds,row_inds,col_inds,:]=talbedos
				albedos=albedos.cpu().numpy().astype(np.uint8)
			else:
				albedos=None
			if len(tcorrects):
				tcorrects=torch.cat(tcorrects,dim=0)
				# print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
				tcorrects=torch.clamp((tcorrects/2.+0.5)*255.,min=0.,max=255.)
				corrects=torch.ones_like(gtCs)*255.
				corrects[batch_inds,row_inds,col_inds,:]=tcorrects
				corrects=corrects.cpu().numpy().astype(np.uint8)
			else:
				corrects=None

			gtcolors=((gtCs/2.+0.5)*255.).cpu().numpy().astype(np.uint8)
			for ind,(color,gtcolor) in enumerate(zip(colors,gtcolors)):
				cv2.imwrite(osp.join(self.root,'rgb%d.png'%ind),color)
				cv2.imwrite(osp.join(self.root,'albedo%d.png'%ind),albedos[ind]) if albedos is not None else None
				cv2.imwrite(osp.join(self.root,'correct%d.png'%ind),corrects[ind]) if corrects is not None else None
				# cv2.imwrite(osp.join(self.root,'light%d.png'%ind),lights[ind]) if lights is not None else None
				cv2.imwrite(osp.join(self.root,'gtrgb%d.png'%ind),gtcolor)

			self.draw=False


	def forward(self,gtCs,gtMs,gtAs,sample_pix,ratio,frame_ids,root=None,**kwargs):
		device=gtCs.device
		# rebuild the computation graph from dataset cpu to gpu
		focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
		cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
		self.maskRender.rasterizer.cameras=cameras
		if self.pcRender:
			self.pcRender.rasterizer.cameras=cameras
		self.info={}
		self.root=None				
		if self.TmpVs is None or self.Tmpfs is None or self.forward_time%self.remesh_intersect==0:		
			self.update_hierarchical_config(device)
			self.TmpVs,self.Tmpfs=self.discretizeSDF(ratio,None,-self.sdfShrinkRadius)
			if self.TmpVs.shape[0]==0:
				print('tmp sdf vanished...')
				assert(False)
			self.remesh_time=1.+np.floor(self.remesh_time)
			self.TmpVs.requires_grad=True
			if self.pcRender is None:
				self.TmpOptimizer=torch.optim.SGD([self.TmpVs], lr=0.5, momentum=0.9)
			else:
				self.TmpOptimizer=torch.optim.SGD([self.TmpVs], lr=0.05, momentum=0.9)				
				self.TmpOptimizer2=None
				# self.TmpOptimizer2=torch.optim.Adam([p for p in self.deformer.parameters() if p.requires_grad], lr=1.e-5)
			tm=om.TriMesh(self.TmpVs.detach().cpu().numpy(),self.Tmpfs.detach().cpu().numpy())
			TmpFid=torch.from_numpy(tm.vertex_face_indices().astype(np.int64)).to(device)
			TmpVid=torch.arange(0,TmpFid.shape[0]).view(-1,1).repeat(1,TmpFid.shape[1]).to(device)
			sel=TmpFid>=0
			self.TmpFid=TmpFid[sel].view(-1)
			self.TmpVid=TmpVid[sel].view(-1)
			self.root=root
			
		assert(abs(self.sdfShrinkRadius)<1.e-8) #check not use shrink mode

		oldTmpVs=self.TmpVs.detach().clone() if self.root else None
		# oldTmpVs=self.TmpVs.detach().clone()
		TmpVnum=self.TmpVs.shape[0]

		N=gtCs.shape[0]
		Rs,trans,rendcond=self.dataset.get_grad_parameters(frame_ids,device)
		defTmpVs=self.deformer(self.TmpVs[None,:,:].expand(N,-1,3),[Rs,trans],ratio=ratio)
		defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[self.Tmpfs for _ in range(N)])
		
		batch_inds=None
		assert(self.pcRender is not None)
		
		self.info['pc_loss']={}
		with torch.no_grad():
			_,frags=self.maskRender(defMeshes)
			batch_inds,row_inds,col_inds,initTmpPs,front_face_ids=utils.FindSurfacePs(self.TmpVs.detach(),self.Tmpfs,frags)
		if hasattr(self.netRender,'albedo') and False:
			temp=self.netRender.albedo(self.TmpVs,1.).view(-1,3)
			features=[torch.cat([temp,torch.ones(self.TmpVs.shape[0],1,device=self.TmpVs.device)],dim=-1)]*N
			# features=[torch.ones(self.TmpVs.shape[0],1,device=self.TmpVs.device)]*N
		else:
			features=[torch.ones(self.TmpVs.shape[0],1,device=self.TmpVs.device) for _ in range(N)]
			
		masks,frags=self.pcRender(Pointclouds(points=defMeshes.verts_list(),features=features))
		radius=self.pcRender.rasterizer.raster_settings.radius
		radius=int(np.round(radius/2.*float(min(H,W))/1.2))
		if radius>0:
			mgtMs=torch.nn.functional.max_pool2d(gtMs,kernel_size=2*radius+1,stride=1,padding=radius)
			total_loss=self.computeTmpPcLoss(defMeshes,[Rs,trans],masks,mgtMs,gtAs,ratio,frags)
		else:
			mgtMs=None
			total_loss=self.computeTmpPcLoss(defMeshes,[Rs,trans],masks,gtMs,gtAs,ratio,frags)
		

		self.save_debug(oldTmpVs,self.Tmpfs,defMeshes,masks,gtMs,mgtMs,gtAs,gtCs,batch_inds,row_inds,col_inds,initTmpPs,[Rs.detach(),trans.detach()],rendcond,ratio)


		sel=gtMs[batch_inds,row_inds,col_inds]>0. #color loss only compute render mask and gt mask intersected part
		batch_inds=batch_inds[sel]
		row_inds=row_inds[sel]
		col_inds=col_inds[sel]
		initTmpPs=initTmpPs[sel]

		pnum=batch_inds.shape[0]
		if pnum>sample_pix*N:
			sel=torch.rand(pnum)<float(sample_pix*N)/float(pnum)
			sel=sel.to(batch_inds.device)
			batch_inds=batch_inds[sel]
			row_inds=row_inds[sel]
			col_inds=col_inds[sel]
			initTmpPs=initTmpPs[sel]
			pnum=batch_inds.shape[0]
		
		cameras=self.maskRender.rasterizer.cameras
		rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())
		#if ang thred set too small, will not converge. 0.08 for this camera setting, may lead to 1 pix error
		#if resolution reach 1000, 0.04 is match 1 pix error, which is more hard to optmize
		#may recompute the ray and pix coords to eliminate the errors
		#after rectified the perspective camera, the error can be very low and keep an relative high converge ratio
		Rs,trans,rendcond=self.dataset.get_grad_parameters(frame_ids,device)
		defconds=[Rs,trans]
		initTmpPs,check=utils.OptimizeSurfacePs(cameras.cam_pos().detach(),rays.detach(),initTmpPs,batch_inds,self.sdf,ratio,self.deformer,defconds,dthreshold=5.e-5,athreshold=self.angThred,w1=3.05,w2=1.,times=10)
		# rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds_.view(-1,1),torch.ones_like(col_inds_.view(-1,1))],dim=-1).float())
		self.info['rayInfo']=(check.numel(),check.sum().item())
		self.TmpPs=None
		# decrease sample to save memory
		nonmnfld_pnts=utils.sample_points(torch.cat([initTmpPs,self.TmpVs[torch.rand(TmpVnum)<4096./float(TmpVnum)].detach()],dim=0), 1.8,0.01)
		nonmnfld_pnts.requires_grad_()
		nonmnfld_pred = self.sdf(nonmnfld_pnts,ratio)
		nonmnfld_grad=self.sdf.gradient(nonmnfld_pnts,nonmnfld_pred)
		grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
		self.info['grad_loss']=grad_loss.item()
		total_loss+=grad_loss*self.conf.get_float('grad_weight')


		nonmnfld_pnts=None





		

		self.info['color_loss']=-1.0
		if self.conf.get_float('color_weight')>0. and self.info['rayInfo'][1]>0:
			self.TmpPs=initTmpPs[check]
			self.TmpPs.requires_grad=True
			self.rays=rays[check]
			self.batch_inds=batch_inds[check]
			self.col_inds=col_inds[check]
			self.row_inds=row_inds[check]
			if not self.idr_like_render:
				nx=utils.compute_deformed_normals(self.sdf,self.deformer,self.TmpPs,defconds,self.batch_inds,ratio,'train')[0]
				colors=self.netRender(self.TmpPs, nx, -self.rays, (self.sdf.rendcond if self.idr_like_cond else rendcond[self.batch_inds]) ,ratio)
			else:
				sdfs=self.sdf(self.TmpPs,ratio)
				nx=torch.autograd.grad(sdfs,self.TmpPs,torch.ones_like(sdfs),retain_graph=True,create_graph=True)[0]
				nx=nx/nx.norm(dim=1,keepdim=True)
				crays=utils.compute_cardinal_rays(self.deformer,self.TmpPs,self.rays,defconds,self.batch_inds,ratio,'train')[0]
				colors=self.netRender(self.TmpPs, nx, -crays, (self.sdf.rendcond if self.idr_like_cond else rendcond[self.batch_inds]),ratio)

			color_loss=(gtCs[self.batch_inds,self.row_inds,self.col_inds,:]-colors).abs().sum(1)
			color_loss=scatter(color_loss,self.batch_inds,reduce='mean',dim_size=N).mean()
			
			self.info['color_loss']=color_loss.item()
			total_loss+=self.conf.get_float('color_weight')*color_loss

			if hasattr(self.netRender,'albedo'):
				albedos=self.netRender.albedo(self.TmpPs,1.).view(-1,3)
				albedo_loss=(gtAs[self.batch_inds,self.row_inds,self.col_inds,:]-albedos).abs().sum(1)
				albedo_loss=scatter(albedo_loss,self.batch_inds,reduce='mean',dim_size=N).mean()				
				self.info['albedo_loss']=albedo_loss.item()
				total_loss+=albedo_loss*(self.conf.get_float('albedo_weight') if 'albedo_weight' in self.conf else self.conf.get_float('color_weight'))



		self.remesh_time=np.floor(self.remesh_time)+float(self.forward_time%self.remesh_intersect)/float(self.remesh_intersect)
		self.info['remesh']=self.remesh_time
		self.forward_time+=1
		return total_loss

	def computeTmpPcLoss(self,defMeshes,defconds,imgs,gtMs,gtAs,ratio,frags):
		N=gtMs.shape[0]
		masks=imgs[...,-1]
		mask_loss=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).mean()
		self.info['pc_loss']['mask_loss']=mask_loss.item()
		loss=mask_loss*(self.conf.get_float('pc_weight.mask_weight') if 'pc_weight.mask_weight' in self.conf else 1.)
		if imgs.shape[-1]>=4:
			maskAs=gtMs.unsqueeze(-1)*gtAs
			albedo_loss=((imgs[...,0:3]-maskAs).abs().reshape(N,-1).sum(-1)/(gtMs.view(N,-1).sum(-1))).sum()/float(N)
			self.info['pc_loss']['albedo_loss']=albedo_loss.item()
			loss=loss+albedo_loss*(self.conf.get_float('pc_weight.albedo_weight') if 'pc_weight.albedo_weight' in self.conf else 10.)
		
		tmpMesh=Meshes(verts=[self.TmpVs],faces=[self.Tmpfs])
		lap_weight=self.conf.get_float('pc_weight.laplacian_weight') if 'pc_weight' in self.conf else -1.
		if lap_weight>0.:
			lap_loss=lap_weight*mesh_laplacian_smoothing(tmpMesh,method='uniform')
			loss=loss+lap_loss
			self.info['pc_loss']['lap_loss']=lap_loss.item()/lap_weight
		edge_weight=self.conf.get_float('pc_weight.edge_weight') if 'pc_weight' in self.conf else -1.
		if edge_weight>0.:
			edge_loss=edge_weight*mesh_edge_loss(tmpMesh,target_length=0.)
			loss=loss+edge_loss
			self.info['pc_loss']['edge_loss']=edge_loss.item()/edge_weight
		norm_weight=self.conf.get_float('pc_weight.norm_weight') if 'pc_weight' in self.conf else -1.
		if norm_weight>0.:
			norm_loss=norm_weight*mesh_normal_consistency(tmpMesh)
			loss=loss+norm_loss
			self.info['pc_loss']['norm_loss']=norm_loss.item()/norm_weight


		if self.only_surface_ps:
			firstPsIds=frags.idx[...,0].long()
			selPsMask=firstPsIds>=0
			index=firstPsIds[selPsMask]
			visible_def_pids=scatter(torch.ones_like(index),index,reduce='sum',dim_size=self.TmpVs.shape[0]*N)>0

			index=(firstPsIds%self.TmpVs.shape[0])[selPsMask]
			# print(index.shape)
			# print(index.dtype)
			visible_tmp_pids=scatter(torch.ones_like(index),index,reduce='sum',dim_size=self.TmpVs.shape[0])>0

		
		self.TmpOptimizer.zero_grad()
		self.TmpOptimizer2.zero_grad() if self.TmpOptimizer2 else None
		loss.backward()
		self.TmpOptimizer.step()
		self.TmpOptimizer2.step() if self.TmpOptimizer2 else None

		if self.only_surface_ps:
			mnfld_pred=self.sdf(self.TmpVs[visible_tmp_pids],ratio)
		else:
			mnfld_pred=self.sdf(self.TmpVs,ratio)
		sdf_loss=(mnfld_pred+self.sdfShrinkRadius).abs().mean()
		self.info['pc_loss_sdf']=sdf_loss.item()
		loss=sdf_loss*(self.conf.get_float('pc_weight.weight') if 'pc_weight' in self.conf else 60.)


		return loss

	# this func must be invoked after forward() getting loss and has run loss.backward()
	# this func propagates grad(TmpPs) to sdf, deformer and camera parameters and conds grad(if requires_grad)
	# to do: check whether torch Function can handle this problem, that will simplify the use
	def propagateTmpPsGrad(self,frame_ids,ratio):
		if self.TmpPs is None or self.TmpPs.grad is None:
			self.info['invInfo']=(-1,-1)
			return
		device=self.TmpPs.device
		Rs,trans,_=self.dataset.get_grad_parameters(frame_ids,device)
		defconds=[Rs,trans]
		focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
		self.maskRender.rasterizer.cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)

		grad_l_p=self.TmpPs.grad
		# to make v can backward to camera parameters
		# after loss.backward(), the computation graph of self.rays to camera params has been destroyed, need to be recomputed
		if self.rays.requires_grad:
			v=self.maskRender.rasterizer.cameras.view_rays(torch.cat([self.col_inds.view(-1,1),self.row_inds.view(-1,1),torch.ones_like(self.col_inds.view(-1,1))],dim=-1).float())
		else:
			v=self.rays.detach()
		c=self.maskRender.rasterizer.cameras.cam_pos()
		p=self.TmpPs #[N,3]
		f=self.sdf(p,ratio)
		grad_f_p=torch.autograd.grad(f,p,torch.ones_like(f),retain_graph=False)[0] #[N,3]
		d=self.deformer(p,defconds,self.batch_inds,ratio=ratio) #[N,3]
		inputs=[p]
		grad_d_p=[]
		opt_defconds=[]
		for defcond in defconds:
			if type(defcond) is list:
				for defc in defcond:
					if defc.requires_grad:
						opt_defconds.append(defc)
			else:
				if defcond.requires_grad:
					opt_defconds.append(defcond)
		grad_outputs=torch.ones_like(d[...,0])
		outx=torch.autograd.grad(d[...,0],inputs,grad_outputs,retain_graph=True)
		grad_d_p.append(outx[0].view(-1,1,3))
		# if len(opt_defconds):
		# 	grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outx[1:],opt_defconds)])
		outy=torch.autograd.grad(d[...,1],inputs,grad_outputs,retain_graph=True)
		grad_d_p.append(outy[0].view(-1,1,3))
		# if len(opt_defconds):
		# 	grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outy[1:],opt_defconds)])
		outz=torch.autograd.grad(d[...,2],inputs,grad_outputs,retain_graph=False)
		grad_d_p.append(outz[0].view(-1,1,3))
		# if len(opt_defconds):
		# 	grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outz[1:],opt_defconds)])
		grad_d_p=torch.cat(grad_d_p,dim=1) #N,3,3
		# if len(opt_defconds):
		# 	grad_d_z=[torch.cat([gradx,grady,gradz],dim=1) for gradx,grady,gradz in zip(grad_d_z[0],grad_d_z[1],grad_d_z[2])]


		v_cross=torch.zeros_like(grad_d_p)
		v_cross[:,0,1]=-v[:,2]
		v_cross[:,0,2]=v[:,1]
		v_cross[:,1,0]=v[:,2]
		v_cross[:,1,2]=-v[:,0]
		v_cross[:,2,0]=-v[:,1]
		v_cross[:,2,1]=v[:,0]
		v_cross=v_cross.detach()
		a1=v_cross.matmul(grad_d_p)
		b=torch.cat([grad_f_p.view(-1,1,3),a1],dim=1)
		btb=b.permute(0,2,1).matmul(b)
		btb_inv,check=Fast3x3Minv(btb)
		self.info['invInfo']=(check.numel(),check.sum().item())
		rhs_1=btb_inv.matmul(b.permute(0,2,1)) #N,3,4
		rhs_1=grad_l_p.view(-1,1,3).matmul(rhs_1) #N,1,4
		loss=0.
		#for theta
		params=[param for param in self.sdf.parameters() if param.requires_grad]
		params_grads=torch.autograd.grad(self.sdf(p,ratio),params,-rhs_1[:,:,0])
		for param,grad in zip(params,params_grads):			
			loss+=(param*grad).sum()
		#for phi
		params=[param for param in self.deformer.parameters() if param.requires_grad]
		d=self.deformer(p,defconds,self.batch_inds,ratio=ratio) #[N,3]
		temp=(rhs_1[:,:,-3:].matmul(-v_cross)).view(-1,3)
		if len(opt_defconds):			
			params_grads=torch.autograd.grad(d,params,temp,retain_graph=True)
		else:
			params_grads=torch.autograd.grad(d,params,temp,retain_graph=False)
		for param,grad in zip(params,params_grads):			
			loss+=(param*grad).sum()
		#for z
		if len(opt_defconds):
			params_grads=torch.autograd.grad(d,opt_defconds,temp,retain_graph=False)
			for opt_defcond,grad in zip(opt_defconds,params_grads):				
				loss+=(opt_defcond*grad).sum()

		if v.requires_grad:
			dc=d.detach()-c.detach().view(1,3)
			dc_cross=torch.zeros_like(grad_d_p)
			dc_cross[:,0,1]=-dc[:,2]
			dc_cross[:,0,2]=dc[:,1]
			dc_cross[:,1,0]=dc[:,2]
			dc_cross[:,1,2]=-dc[:,0]
			dc_cross[:,2,0]=-dc[:,1]
			dc_cross[:,2,1]=dc[:,0]
			grad=rhs_1[:,:,-3:].matmul(dc_cross).view(-1,3)
			# to do: how to propagate v and c grads to camera parameters?			
			loss+=(v*grad).sum()

		if c.requires_grad:
			grad=-temp.sum(0)
			loss+=(c*grad).sum()
		loss.backward()

from .network import getTmpSdf
from .Deformer import RigidTransformer

def getRigidOptNet(dataset,N,resolutions,device,conf,use_initial_sdf=True,use_initial_skinner=True):

	sdf_multires=conf.get_int('sdf_net.multires')
	tmpSdf=getTmpSdf(device,sdf_multires,0.6,(256 if conf.get_bool('train.idr_like_cond') else 0))
	if osp.isfile(osp.join(dataset.root,'initial_sdf'+('_idr' if conf.get_bool('train.idr_like_cond') else '')+'_%d.pth'%sdf_multires)) and use_initial_sdf:
		tmpSdf.load_state_dict(torch.load(osp.join(dataset.root,'initial_sdf'+('_idr' if conf.get_bool('train.idr_like_cond') else '')+'_%d.pth'%sdf_multires),map_location='cpu'))
	sdf_initialized=-1
	
	deformer=RigidTransformer().to(device)
	cam_data=dataset.camera_params
	cameras=RectifiedPerspectiveCameras(cam_data['focal_length'].view(1,2).expand(N,2),cam_data['princeple_points'].view(1,2).expand(N,2),
				utils.quat2mat(cam_data['cam2world_coord_quat'].view(1,4)).expand(N,3,3),cam_data['world2cam_coord_trans'].view(1,3).expand(N,3),
				image_size=[(dataset.W, dataset.H)]).to(device)
	bmins=[-0.8,-1.25,-0.4]
	bmaxs=[0.8,1.0,0.4]
	engine = Seg3dLossless(
			query_func=None, 
			b_min = bmins,
			b_max = bmaxs,
			resolutions=resolutions,
			align_corners=False,
			balance_value=0.0, # be careful
			device=device, 
			visualize=False,
			debug=False,
			use_cuda_impl=False,
			faster=False 
		)
	sigma = conf.get_float('train.coarse.sigma')
	# sigma = 1.e-5
	raster_settings_silhouette = RasterizationSettings(
	image_size=(dataset.H,dataset.W), 
	blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
	# bin_size=0,
	faces_per_pixel=conf.get_int('train.coarse.faces_per_pixel'),
	perspective_correct=True,
	clip_barycentric_coords=False,
	cull_backfaces=False
	)	
	renderer = MeshRendererWithFragments(
	rasterizer=MeshRasterizer(
		cameras=cameras, 
		raster_settings=raster_settings_silhouette
	),
	shader=SoftSilhouetteShader(BlendParams(sigma=sigma))
	)
	rendnet=RenderNet.getRenderNet(device,conf.get_config('render_net'))
	# rendnet=getattr(RenderNet,conf.get_string('render_net.type'))(conf.get_int('render_net.condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires=conf.get_int('render_net.multires')).to(device)
	optNet=OptimRigidNetwork(tmpSdf,deformer,engine,renderer,rendnet,conf=conf.get_config('loss_coarse'))
	optNet.remesh_intersect=conf.get_int('train.coarse.remesh_intersect')
	optNet=optNet.to(device)
	optNet.dataset=dataset
	if hasattr(rendnet,'albedo'):
		optNet.dataset.require_albedo=True
	if 'train.only_surface_ps' in conf:
		optNet.only_surface_ps=conf.get_bool('train.only_surface_ps')
	if 'train.idr_like_render' in conf:
		optNet.idr_like_render=conf.get_bool('train.idr_like_render')
	if 'train.idr_like_cond' in conf:
		optNet.idr_like_cond=conf.get_bool('train.idr_like_cond')

	return optNet, sdf_initialized