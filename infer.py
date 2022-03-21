import torch
import numpy as np
from dataset.dataset import getDatasetAndLoader
from model import getOptNet
from pyhocon import ConfigFactory,HOCONConverter
import argparse
import trimesh
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

parser = argparse.ArgumentParser(description='neu video body infer')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
					help='gpu ids')
parser.add_argument('--batch-size',default=1,type=int,metavar='IDs',
					help='batch size')
# parser.add_argument('--conf',default=None,metavar='M',
# 					help='config file')
parser.add_argument('--rec-root',default=None,metavar='M',
					help='data root')
parser.add_argument('--nV',action='store_true',help='not save video')
parser.add_argument('--nI',action='store_true',help='not save image')
parser.add_argument('--C',action='store_true',help='overlay on gtimg')
parser.add_argument('--nColor',action='store_true',help='not render images')
# parser.add_argument('--model',default=None,metavar='M',
# 					help='pretrained scene model')
# parser.add_argument('--save-folder',default=None,metavar='M',help='save folder')
args = parser.parse_args()

assert(not(args.nV and args.nI))

resolutions = [
	(32+1, 32+1, 32+1),
	(64+1, 64+1, 64+1),
	(128+1, 128+1, 128+1),
	(256+1, 256+1, 256+1),
	(512+1, 512+1, 512+1),
]

config=ConfigFactory.parse_file(osp.join(args.rec_root,'config.conf'))
device=args.gpu_ids[0]
deformer_condlen=config.get_int('mlp_deformer.condlen')
renderer_condlen=config.get_int('render_net.condlen')
# batch_size=config.get_int('train.coarse.batch_size')
batch_size=args.batch_size
shuffle=False
dataset,dataloader=getDatasetAndLoader(osp.normpath(osp.join(args.rec_root,osp.pardir)),{'deformer':deformer_condlen,'renderer':renderer_condlen},batch_size,
						shuffle,config.get_int('train.num_workers'),
						False,False,False)

optNet,sdf_initialized=getOptNet(dataset,batch_size,None,None,resolutions,device,config)

print('load model: '+osp.join(args.rec_root,'latest.pth'))
optNet,dataset=utils.load_model(osp.join(args.rec_root,'latest.pth'),optNet,dataset,device)
optNet.dataset=dataset
optNet.eval()

raster_settings = RasterizationSettings(
	image_size=(dataset.H,dataset.W), 
	blur_radius=0, 
	faces_per_pixel=1,
	perspective_correct=True,
	clip_barycentric_coords=False,
	cull_backfaces=False
	)

optNet.maskRender.rasterizer.raster_settings=raster_settings
optNet.maskRender.shader=HardPhongShader(device,optNet.maskRender.rasterizer.cameras)
optNet.pcRender=None
H=dataset.H
W=dataset.W
if 'train.fine.point_render' in config:
	raster_settings_silhouette = PointsRasterizationSettings(
		image_size=(H,W), 
		radius=config.get_float('train.fine.point_render.radius'),
		# radius=0.002,
		bin_size=64,
		points_per_pixel=50,
		)   
	optNet.pcRender=PointsRenderer(
		rasterizer=PointsRasterizer(
			cameras=optNet.maskRender.rasterizer.cameras, 
			raster_settings=raster_settings_silhouette
		),
			compositor=AlphaCompositor(background_color=(1,1,1,1))
		).to(device)






ratio={'sdfRatio':1.,'deformerRatio':1.,'renderRatio':1.}
TmpVs,Tmpfs=optNet.discretizeSDF(ratio,None,0.)

mesh = trimesh.Trimesh(TmpVs.detach().cpu().numpy(), Tmpfs.cpu().numpy())
mesh.export(osp.join(args.rec_root,'tmp.ply'))

if not osp.isdir(osp.join(args.rec_root,'colors')):
	os.makedirs(osp.join(args.rec_root,'colors'))
if not osp.isdir(osp.join(args.rec_root,'meshs')):
	os.makedirs(osp.join(args.rec_root,'meshs'))
if not osp.isdir(osp.join(args.rec_root,'def1meshs')):
	os.makedirs(osp.join(args.rec_root,'def1meshs'))
if not args.nV:
	writer_meshs=cv2.VideoWriter(osp.join(args.rec_root,'meshs/video.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30.,(W,H))
	writer_def1meshs=cv2.VideoWriter(osp.join(args.rec_root,'def1meshs/video.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30.,(W,H))
	writer_colors=None
	writer_albedos=None
	writer_pcmasks=None
errors={}
errors['maskE']=-1.*np.ones((len(dataset)))
gts={}
# for data_index, (frame_ids, imgs, masks, albedos) in enumerate(dataloader):
for data_index, (frame_ids, outs) in enumerate(dataloader):
	imgs=outs['img']
	masks=outs['mask']
	if 'albedo' in outs:
		albedos=outs['albedo']
	else:
		albedos=None
	if args.nColor:
		print(data_index*batch_size)
	else:
		print(data_index*batch_size,end='	')
	frame_ids=frame_ids.long().to(device)
	gts['mask']=masks.to(device)
	if args.C:
		gts['image']=(imgs.to(device)+1.)/2.
	colors,albedos,imgs,def1imgs,pcmasks=optNet.infer(TmpVs,Tmpfs,dataset.H,dataset.W,ratio,frame_ids,args.nColor,gts)
	for fid,img,def1img in zip(frame_ids.cpu().numpy().reshape(-1),imgs,def1imgs):
		if not args.nV:
			writer_meshs.write(img[:,:,[2,1,0]])
			writer_def1meshs.write(def1img[:,:,[2,1,0]])
		if not args.nI:
			cv2.imwrite(osp.join(args.rec_root,'meshs/%d.png'%fid),img[:,:,[2,1,0]])
			cv2.imwrite(osp.join(args.rec_root,'def1meshs/%d.png'%fid),def1img[:,:,[2,1,0]])
	if colors is not None:
		os.makedirs(osp.join(args.rec_root,'colors')) if not osp.isdir(osp.join(args.rec_root,'colors')) else None
		if not args.nV and writer_colors is None:
			writer_colors=cv2.VideoWriter(osp.join(args.rec_root,'colors/video.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30.,(W,H))
		if not args.nI:			
			for fid,color in zip(frame_ids.cpu().numpy().reshape(-1),colors):
				writer_colors.write(color) if not args.nV else None
				cv2.imwrite(osp.join(args.rec_root,'colors/%d.png'%fid),color)
	if albedos is not None:
		os.makedirs(osp.join(args.rec_root,'albedos')) if not osp.isdir(osp.join(args.rec_root,'albedos')) else None
		if not args.nV and writer_albedos is None:
			writer_albedos=cv2.VideoWriter(osp.join(args.rec_root,'albedos/video.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30.,(W,H))
		if not args.nI:			
			for fid,albedo in zip(frame_ids.cpu().numpy().reshape(-1),albedos):
				writer_albedos.write(albedo) if not args.nV else None
				cv2.imwrite(osp.join(args.rec_root,'albedos/%d.png'%fid),albedo)
	if pcmasks is not None:
		os.makedirs(osp.join(args.rec_root,'pcmasks')) if not osp.isdir(osp.join(args.rec_root,'pcmasks')) else None
		if not args.nV and writer_pcmasks is None:
			writer_pcmasks=cv2.VideoWriter(osp.join(args.rec_root,'pcmasks/video.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30.,(W,H))
		if not args.nI:			
			for fid,pcmask in zip(frame_ids.cpu().numpy().reshape(-1),pcmasks):
				writer_pcmasks.write(pcmask[:,:,[2,1,0]]) if not args.nV else None
				cv2.imwrite(osp.join(args.rec_root,'pcmasks/%d.png'%fid),pcmask)
	errors['maskE'][frame_ids.cpu().numpy()]=gts['maskE']
	# assert(False)

if not args.nV:
	writer_meshs.release()
	writer_def1meshs.release()
	if writer_colors:
		writer_colors.release()
	if writer_albedos:
		writer_albedos.release()
	if writer_pcmasks:
		writer_pcmasks.release()

with open(osp.join(args.rec_root,'errors.txt'),'w') as ff:
	ff.write('      mask\n')
	maskE=errors['maskE']
	for ind,e in enumerate(maskE.tolist()):
		if e>=0.:
			ff.write('%4d: %.4f\n'%(ind,e))
	maskE=maskE[maskE>=0.]
	ff.write('mask mean: %.4f, max: %.4f, min: %.4f, maxinds:'%(maskE.mean(),maskE.max(),maskE.min()))
	for ind in (-maskE).argsort()[:10]:
		ff.write('%d '%ind)

# dataset.poses=smooth_poses.view(-1,24,3)
# dataset.trans=smooth_trans.view(-1,3)

# writer_color=cv2.VideoWriter(osp.join(args.rec_root,'colors/video_smooth.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30.,(1080,1080))
# writer_meshs=cv2.VideoWriter(osp.join(args.rec_root,'meshs/video_smooth.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30.,(1080,1080))
# for data_index, (frame_ids, imgs, masks) in enumerate(dataloader):
# 	print(data_index*batch_size,end='	')
# 	if data_index==250//batch_size:
# 		break
# 	frame_ids=frame_ids.long().to(device)
# 	colors,imgs=optNet.infer(TmpVs,Tmpfs,dataset.H,dataset.W,1.,frame_ids)
# 	for fid,color,img in zip(frame_ids.cpu().numpy().reshape(-1),colors,imgs):
# 		writer_color.write(color)
# 		writer_meshs.write(img[:,:,[2,1,0]])
# 		# cv2.imwrite(osp.join(args.rec_root,'colors/%d.png'%fid),color)
# 		# cv2.imwrite(osp.join(args.rec_root,'meshs/%d.png'%fid),img[:,:,[2,1,0]])
# 	# assert(False)
# writer_color.release()
# writer_meshs.release()

print('done')

