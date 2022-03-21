import numpy as np
import torch
import os
import cv2
import os.path as osp
from glob import glob
import utils

class RigidDataset(torch.utils.data.Dataset):
	# get a batch_size continuous frame sequence
	def __init__(self,data_root,conds_lens={}):
		self.root=data_root
		self.read_data()
		self.require_albedo=False
		self.conds=[]
		self.cond_ns=[]
		# cond parameter needs optimization by default
		for name,length in conds_lens.items():
			# cond=torch.zeros(self.frame_num,length,requires_grad=True)			
			# torch.nn.init.normal_(cond, mean=0., std=0.001)
			cond=((0.1*torch.randn(length,self.frame_num//5)).matmul(utils.DCTSpace(self.frame_num//5,self.frame_num))).transpose(0,1)
			cond.requires_grad_()
			self.conds.append(cond)
			self.cond_ns.append(name)

	def read_data(self):
		candidate_ext=['.jpg','.png']
		imgs=[]
		for ext in candidate_ext:
			imgs.extend(glob(osp.join(self.root,'imgs/*'+ext)))
		imgs.sort(key=lambda x: int(osp.basename(x).split('.')[0]))
		self.frame_num=len(imgs)
		self.img_ns=imgs

		masks=[]
		for ext in candidate_ext:
			masks.extend(glob(osp.join(self.root,'masks/*'+ext)))
		masks.sort(key=lambda x: int(osp.basename(x).split('.')[0]))
		self.mask_ns=masks
		assert(len(self.mask_ns)==self.frame_num)
		for ind,(img_n,mask_n) in enumerate(zip(self.img_ns,self.mask_ns)):
			assert(ind==int(osp.basename(img_n).split('.')[0]))
			assert(ind==int(osp.basename(mask_n).split('.')[0]))
			
		self.H,self.W,_=cv2.imread(self.mask_ns[0]).shape
		data=np.load(osp.join(self.root,'frame_RTs.npz'))
		self.Rs=torch.from_numpy(data['Rs'].astype(np.float32)).view(-1,3,3)
		self.trans=torch.from_numpy(data['Ts'].astype(np.float32)).view(-1,3)
		data=np.load(osp.join(self.root,'camera.npz'))
		self.camera_params={'focal_length':torch.tensor(np.array([data['fx'],data['fy']]).astype(np.float32)), \
							'princeple_points':torch.tensor(np.array([data['cx'],data['cy']]).astype(np.float32)), \
							'cam2world_coord_quat':torch.from_numpy(data['quat'].astype(np.float32)).view(-1), \
							'world2cam_coord_trans':torch.from_numpy(data['T'].astype(np.float32)).view(-1)}
	def opt_camera_params(self,conf):
		if type(conf)==bool:
			self.camera_params['focal_length'].requires_grad_(conf)
			self.camera_params['princeple_points'].requires_grad_(conf)
			self.camera_params['cam2world_coord_quat'].requires_grad_(conf)
			self.camera_params['world2cam_coord_trans'].requires_grad_(conf)
		else:
			self.camera_params['focal_length'].requires_grad_(conf.get_bool('focal_length'))
			self.camera_params['princeple_points'].requires_grad_(conf.get_bool('princeple_points'))
			self.camera_params['cam2world_coord_quat'].requires_grad_(conf.get_bool('quat'))
			self.camera_params['world2cam_coord_trans'].requires_grad_(conf.get_bool('T'))

	def learnable_weights(self):
		ws=[]
		ws.extend([cond for cond in self.conds if cond.requires_grad])
		ws.extend([v for k,v in self.camera_params.items() if v.requires_grad])
		ws.extend([v for v in [self.Rs,self.trans] if v.requires_grad])
		return ws

	def __len__(self):
		return self.frame_num
	def __getitem__(self, idx):
		# convert to [-1.,1.] to keep consistent with render net tanh output
		img=torch.from_numpy((cv2.imread(self.img_ns[idx]).astype(np.float32)/255.-0.5)*2).view(self.H,self.W,3)
		mask=(torch.from_numpy(cv2.imread(self.mask_ns[idx]))>0).view(self.H,self.W,-1).any(-1).float()
		if self.require_albedo:
			albedo=torch.from_numpy((cv2.imread(osp.join(self.root,'albedos/%d.png'%idx)).astype(np.float32)/255.-0.5)*2.).view(self.H,self.W,3)
		else:
			albedo=1.
		return idx,img,mask,albedo
	# this function is a patch for __getitem__, it seems that dataloader cannot fetch data with requires_grad=True, because of stack(out=out)
	def get_grad_parameters(self,idxs,device):
		conds=[cond[idxs].to(device) for cond in self.conds]
		if len(conds):
			return self.Rs[idxs].to(device),self.trans[idxs].to(device),*conds
		else:
			return self.Rs[idxs].to(device),self.trans[idxs].to(device),None

	def get_camera_parameters(self,N,device):
		return (self.camera_params['focal_length'].to(device).view(1,2).expand(N,2),self.camera_params['princeple_points'].to(device).view(1,2).expand(N,2), \
			utils.quat2mat(self.camera_params['cam2world_coord_quat'].to(device).view(1,4)).expand(N,3,3),self.camera_params['world2cam_coord_trans'].to(device).view(1,3).expand(N,3),self.H,self.W)

class SceneDataset(torch.utils.data.Dataset):
	# get a batch_size continuous frame sequence
	def __init__(self,data_root, conds_lens={}):
		self.root=data_root
		self.read_data()
		self.require_albedo=False
		self.conds=[]
		self.cond_ns=[]
		# cond parameter needs optimization by default
		for name,length in conds_lens.items():
			# cond=torch.zeros(self.frame_num,length,requires_grad=True)			
			# torch.nn.init.normal_(cond, mean=0., std=0.001)
			cond=((0.1*torch.randn(length,self.frame_num//5)).matmul(utils.DCTSpace(self.frame_num//5,self.frame_num))).transpose(0,1)
			cond.requires_grad_()
			self.conds.append(cond)
			self.cond_ns.append(name)

	def read_data(self):
		candidate_ext=['.jpg','.png']
		imgs=[]
		for ext in candidate_ext:
			imgs.extend(glob(osp.join(self.root,'imgs/*'+ext)))
		imgs.sort(key=lambda x: int(osp.basename(x).split('.')[0]))
		self.frame_num=len(imgs)
		self.img_ns=imgs
		self.mask_ns=[]
		for ind,img_n in enumerate(self.img_ns):
			assert(ind==int(osp.basename(img_n).split('.')[0]))
			# self.mask_ns.append(osp.join(self.root,'masks/%d.png'%ind))
			self.mask_ns.append(osp.join(self.root,'masks/%s.png'%(osp.basename(img_n).split('.')[0])))
			assert(osp.isfile(self.mask_ns[-1]))
		self.H,self.W,_=cv2.imread(self.mask_ns[0]).shape
		data=np.load(osp.join(self.root,'smpl_rec.npz'))
		self.poses=torch.from_numpy(data['poses'].astype(np.float32)).view(-1,24,3)
		self.trans=torch.from_numpy(data['trans'].astype(np.float32)).view(-1,3)
		self.shape=torch.from_numpy(data['shape'].astype(np.float32)).view(-1)
		self.gender=str(data['gender']) if 'gender' in data else 'neutral'
		print('scene data use %s smpl'%self.gender)
		
		if 'vid_seg_indices' in data:
			if type(data['vid_seg_indices'])==np.ndarray:
				self.video_segmented_index=data['vid_seg_indices'][0:-1].tolist()
			else:
				self.video_segmented_index=data['vid_seg_indices'][0:-1]
			if len(self.video_segmented_index)>0:
				print('this dataset has %d segmented videos'%(len(self.video_segmented_index)+1))
		else:
			self.video_segmented_index=[]

		data=np.load(osp.join(self.root,'camera.npz'))
		self.camera_params={'focal_length':torch.tensor(np.array([data['fx'],data['fy']]).astype(np.float32)), \
							'princeple_points':torch.tensor(np.array([data['cx'],data['cy']]).astype(np.float32)), \
							'cam2world_coord_quat':torch.from_numpy(data['quat'].astype(np.float32)).view(-1), \
							'world2cam_coord_trans':torch.from_numpy(data['T'].astype(np.float32)).view(-1)}
		
	def opt_camera_params(self,conf):
		if type(conf)==bool:
			self.camera_params['focal_length'].requires_grad_(conf)
			self.camera_params['princeple_points'].requires_grad_(conf)
			self.camera_params['cam2world_coord_quat'].requires_grad_(conf)
			self.camera_params['world2cam_coord_trans'].requires_grad_(conf)
		else:
			self.camera_params['focal_length'].requires_grad_(conf.get_bool('focal_length'))
			self.camera_params['princeple_points'].requires_grad_(conf.get_bool('princeple_points'))
			self.camera_params['cam2world_coord_quat'].requires_grad_(conf.get_bool('quat'))
			self.camera_params['world2cam_coord_trans'].requires_grad_(conf.get_bool('T'))

	def learnable_weights(self):
		ws=[]
		ws.extend([cond for cond in self.conds if cond.requires_grad])
		ws.extend([v for k,v in self.camera_params.items() if v.requires_grad])
		ws.extend([v for v in [self.shape,self.poses,self.trans] if v.requires_grad])
		return ws

	def __len__(self):
		return self.frame_num
	def __getitem__(self, idx):
		# convert to [-1.,1.] to keep consistent with render net tanh output
		out={}
		img=torch.from_numpy((cv2.imread(self.img_ns[idx]).astype(np.float32)/255.-0.5)*2).view(self.H,self.W,3)
		out['img']=img
		# rec_f=self.img_ns[idx].replace('.%s' % (self.img_ns[idx].split('.')[-1]), '_rect.txt')		
		# if osp.isfile(rec_f):
		# 	rects=np.loadtxt(rect_f, dtype=np.int32)
		# 	if len(rects.shape) == 1:
		# 		rects = rects[None]
		# 	rect = torch.from_numpy(rects[0].astype(np.float32))
		# 	out['rect']=rect
		mask=(torch.from_numpy(cv2.imread(self.mask_ns[idx]))>0).view(self.H,self.W,-1).any(-1).float()
		out['mask']=mask
		norm_f=self.img_ns[idx].replace('/imgs/','/normals/')[:-3]+'png'
		if osp.isfile(norm_f):
			normals=cv2.imread(norm_f)[:,:,::-1]
			normals=2.*normals.astype(np.float32)/255.-1.
			out['normal']=normals
		# norm_e=self.img_ns[idx].replace('/imgs/','/normal_edges/')[:-3]+'png'
		# if osp.isfile(norm_e):
		# 	normal_edges=cv2.imread(norm_e,cv2.IMREAD_UNCHANGED)

		# 	normal_edges=normal_edges.astype(np.float32)/255.
		# 	out['normal_edge']=normal_edges
		if self.require_albedo:
			albedo=torch.from_numpy((cv2.imread(osp.join(self.root,'albedos/%d.png'%idx)).astype(np.float32)/255.-0.5)*2.).view(self.H,self.W,3)
			out['albedo']=albedo

		# return idx,img,mask,albedo
		return idx,out
	# this function is a patch for __getitem__, it seems that dataloader cannot fetch data with requires_grad=True, because of stack(out=out)
	def get_grad_parameters(self,idxs,device):
		conds=[cond[idxs].to(device) for cond in self.conds]		
		if len(conds)>1:
			return self.poses[idxs].to(device),self.trans[idxs].to(device),*conds
		else:
			return self.poses[idxs].to(device),self.trans[idxs].to(device),*conds, None


	def get_camera_parameters(self,N,device):
		return (self.camera_params['focal_length'].to(device).view(1,2).expand(N,2),self.camera_params['princeple_points'].to(device).view(1,2).expand(N,2), \
			utils.quat2mat(self.camera_params['cam2world_coord_quat'].to(device).view(1,4)).expand(N,3,3),self.camera_params['world2cam_coord_trans'].to(device).view(1,3).expand(N,3),self.H,self.W)
	def get_batchframe_data(self,name,fids,batchsize):
		

		if len(self.video_segmented_index)==0:
			assert(batchsize<self.frame_num)
			assert(hasattr(self,name))
			videodata=getattr(self,name)
			assert(videodata.shape[0]>=self.frame_num)
			videodata=videodata[:self.frame_num].to(fids.device)
			starts=fids-batchsize//2
			ends=starts+batchsize			
			offset=starts-0
			sel=offset<0
			starts[sel]-=offset[sel]
			ends[sel]-=offset[sel]			
			offset=ends-self.frame_num
			sel=offset>0
			starts[sel]-=offset[sel]
			ends[sel]-=offset[sel]
			return videodata[starts.view(-1,1)+torch.arange(0,batchsize).view(1,batchsize).to(fids.device)], fids-starts
		elif len(self.video_segmented_index)==1:
			def extract_batch(start,end,fids,batchsize):
				starts=fids-batchsize//2
				ends=starts+batchsize
				offset=starts-start
				sel=offset<0
				starts[sel]-=offset[sel]
				ends[sel]-=offset[sel]

				offset=ends-end
				sel=offset>0
				starts[sel]-=offset[sel]
				ends[sel]-=offset[sel]
				return starts,ends

			assert(hasattr(self,name))
			videodata=getattr(self,name)
			assert(videodata.shape[0]>=self.frame_num)
			videodata=videodata[:self.frame_num].to(fids.device)
			starts=torch.zeros_like(fids)-1
			ends=torch.zeros_like(fids)-1

			start=0
			end=self.video_segmented_index[0]
			sel=(fids>=start) * (fids<end)
			assert(batchsize<end-start)
			ss,es=extract_batch(start,end,fids[sel],batchsize)
			starts[sel]=ss
			ends[sel]=es

			start=self.video_segmented_index[0]
			end=self.frame_num
			sel=(fids>=start) * (fids<end)
			assert(batchsize<end-start)
			ss,es=extract_batch(start,end,fids[sel],batchsize)
			starts[sel]=ss
			ends[sel]=es
			assert((starts>=0).all().item())
			assert((ends>=0).all().item())
			return videodata[starts.view(-1,1)+torch.arange(0,batchsize).view(1,batchsize).to(fids.device)], fids-starts


		else:
			raise NotImplementedError



import random
class ClipSampler(torch.utils.data.Sampler):
	def __init__(self,data_source,clip_size,shuffle):
		self.data_source=data_source
		self.clip_size=clip_size
		self.shuffle=shuffle
		self.n=len(self.data_source)//self.clip_size
		if len(self.data_source)==self.n*self.clip_size:
			self.n=self.n-1
		self.start=len(self.data_source)-self.n*self.clip_size
	def __iter__(self):
		if self.shuffle:		
			start=random.sample(list(range(0,self.start+1)),1)[0]
		else:
			start=0
		assert(start+self.n*self.clip_size<=len(self.data_source))
		out=torch.arange(start,start+self.n*self.clip_size).view(self.n,self.clip_size)
		if self.shuffle:
			out=out[torch.randperm(self.n)]
		return iter(out.view(-1).tolist())
	def __len__(self):
		return self.n*self.clip_size

class RandomSampler(torch.utils.data.Sampler):
	def __init__(self,data_source,intersect,shuffle):
		self.length=len(data_source)
		self.intersect=intersect
		self.shuffle=shuffle
		self.n=(self.length-1)//self.intersect+1
		self.start=self.length-self.intersect*(self.n-1)
	def __iter__(self):
		# return iter([0,55,85,595,380]*48)
		# return iter([55]*240)
		if self.shuffle:
			start=random.sample(list(range(0,self.start)),1)[0]
			index=torch.arange(start,self.length,self.intersect)
			index=index[torch.randperm(self.n)]
		else:
			index=torch.arange(0,self.length,self.intersect)	
		assert(index.numel()==self.n)
		return iter(index.view(-1).tolist())
	def __len__(self):
		return self.n


def getDatasetAndLoader(root,conds_lens,batch_size,shuffle,num_workers,opt_pose,opt_trans,opt_camera):
	dataset=SceneDataset(root,conds_lens)
	if opt_pose:
		dataset.poses.requires_grad_(True)
	if opt_trans:
		dataset.trans.requires_grad_(True)
	dataset.opt_camera_params(opt_camera)
	# sampler=ClipSampler(dataset,batch_size,shuffle)
	sampler=RandomSampler(dataset,1,shuffle)
	dataloader=torch.utils.data.DataLoader(dataset,batch_size,sampler=sampler,num_workers=num_workers)
	return dataset,dataloader

def getRigidDatasetAndLoader(root,conds_lens,batch_size,shuffle,num_workers,opt_R,opt_trans,opt_camera):
	dataset=RigidDataset(root,conds_lens)
	if opt_R:
		dataset.Rs.requires_grad_(True)
	if opt_trans:
		dataset.trans.requires_grad_(True)
	dataset.opt_camera_params(opt_camera)
	# sampler=ClipSampler(dataset,batch_size,shuffle)
	sampler=RandomSampler(dataset,1,shuffle)
	dataloader=torch.utils.data.DataLoader(dataset,batch_size,sampler=sampler,num_workers=num_workers)
	return dataset,dataloader