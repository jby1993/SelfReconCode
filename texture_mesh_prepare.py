import torch
import numpy as np
from dataset.dataset import getDatasetAndLoader
from model import getOptNet
from pyhocon import ConfigFactory,HOCONConverter
import argparse
import trimesh
import openmesh as om
import os
import os.path as osp
from MCAcc import Seg3dLossless
import utils
import cv2
from tqdm import tqdm
from pytorch3d.renderer import (
	RasterizationSettings, 
	MeshRasterizer,
	SoftSilhouetteShader,
	HardPhongShader,
	BlendParams,
	PointsRasterizationSettings,
	PointsRenderer,
	PointsRasterizer,
	AlphaCompositor
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.io import load_obj
import scipy
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description='neu video body infer')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
					help='gpu ids')
parser.add_argument('--num',default=120,type=int,metavar='IDs',
					help='Number of used frames')
# parser.add_argument('--conf',default=None,metavar='M',
#                   help='config file')
parser.add_argument('--rec-root',default=None,metavar='M',
					help='data root')
args = parser.parse_args()
root=osp.normpath(args.rec_root)
assert(osp.isfile(osp.join(args.rec_root,'template','uvmap.obj')))

config=ConfigFactory.parse_file(osp.join(root,'config.conf'))
device=args.gpu_ids[0]
deformer_condlen=config.get_int('mlp_deformer.condlen')
renderer_condlen=config.get_int('render_net.condlen')
# batch_size=config.get_int('train.coarse.batch_size')
batch_size=1
shuffle=False
dataset,dataloader=getDatasetAndLoader(osp.normpath(osp.join(root,osp.pardir)),{'deformer':deformer_condlen,'renderer':renderer_condlen},batch_size,
						shuffle,config.get_int('train.num_workers'),
						False,False,False)

resolutions = [
  (10+1, 14+1, 6+1),
  (20+1, 28+1, 12+1),
  (40+1, 56+1, 24+1),
  (80+1, 112+1, 48+1),
  (160+1, 224+1, 96+1),
]

optNet,sdf_initialized=getOptNet(dataset,batch_size,None,None,resolutions,device,config)

print('load model: '+osp.join(root,'latest.pth'))
optNet,dataset=utils.load_model(osp.join(root,'latest.pth'),optNet,dataset,device)
optNet.dataset=dataset
optNet.eval()


ratio={'sdfRatio':1.,'deformerRatio':1.,'renderRatio':1.}

tmpMesh=load_obj(osp.join(args.rec_root,'template','uvmap.obj'))

TmpVs=tmpMesh[0].to(device)
Tmpfs=tmpMesh[1].verts_idx.to(device)

TmpTexCoords=tmpMesh[2].verts_uvs.numpy()
TmpTexfs=tmpMesh[1].textures_idx.numpy()

indices_texture = np.ceil(np.arange(args.num) * dataset.frame_num * 1. / args.num).astype(np.int)
# indices_texture = np.ceil(np.arange(args.num) * 260 * 1. / args.num).astype(np.int)

defVs=[]

with torch.no_grad():
	for fid in indices_texture:
		fid=torch.tensor([fid])
		poses,trans,d_cond,rendcond=optNet.dataset.get_grad_parameters(fid,device)
		defTmpVs=optNet.deformer(TmpVs[None,:,:],[d_cond,[poses,trans]],ratio=ratio)
		defVs.append(defTmpVs.cpu().numpy().reshape(-1,3))
	# poses,trans,d_cond,rendcond=optNet.dataset.get_grad_parameters(0,device)
	# inter_results={}
	# optNet.deformer.defs[1](TmpVs.unsqueeze(0),[poses.unsqueeze(0),trans.unsqueeze(0)],inter_results=inter_results)
defVs=np.stack(defVs)

cam_data={}
cam_data['cam_c']=dataset.camera_params['princeple_points'].cpu().view(2).numpy()
cam_data['cam_f']=dataset.camera_params['focal_length'].cpu().view(2).numpy()
cam_data['cam_k']=np.zeros(5)
cam_t=dataset.camera_params['world2cam_coord_trans'].cpu().view(3).numpy()
cam_t[0]=-cam_t[0]
cam_t[1]=-cam_t[1]
cam_data['cam_t']=cam_t
cam_rt=dataset.camera_params['cam2world_coord_quat'].cpu().view(4)[[1,2,3,0]].numpy()
cam_rt/=np.linalg.norm(cam_rt)
cam_rt=Rotation.from_quat(cam_rt).as_matrix().T
cam_rt[0,:]=-cam_rt[0,:]
cam_rt[1,:]=-cam_rt[1,:]
cam_rt=Rotation.from_matrix(cam_rt).as_rotvec()
cam_data['cam_rt']=cam_rt

np.savez(osp.join(args.rec_root,'template','tex_predata.npz'),vt=TmpTexCoords,ft=TmpTexfs,tmpvs=TmpVs.cpu().numpy(),fs=Tmpfs.cpu().numpy(),defVs=defVs,fids=indices_texture,**cam_data)
# np.savez(osp.join(args.rec_root,'template','data.npz'),skin_ws=inter_results['weights'].cpu().numpy(),shape=dataset.shape.cpu().numpy(),pose=utils.smpl_tmp_Apose(config.get_int('train.skinner_pose_type') if 'train.skinner_pose_type' in config else 0))

print('done')