import torch
import numpy as np
from dataset.dataset import getRigidDatasetAndLoader
from model import getRigidOptNet
from pyhocon import ConfigFactory,HOCONConverter
import argparse
import trimesh
import os
import os.path as osp
from MCAcc import Seg3dLossless
import utils
parser = argparse.ArgumentParser(description='neu video body rec')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
					help='gpu ids')
parser.add_argument('--conf',default=None,metavar='M',
					help='config file')
parser.add_argument('--data',default=None,metavar='M',
					help='data root')
parser.add_argument('--model',default=None,metavar='M',
					help='pretrained scene model')
parser.add_argument('--sdf-model',default=None,metavar='M',
					help='substitute sdf model')
parser.add_argument('--save-folder',default=None,metavar='M',help='save folder')
args = parser.parse_args()

# # mask render sigma setting should keep same multipler with mean triangle area
# resolutions={'coarse':
# # [
# # 	(8+1, 14+1, 4+1),
# # 	(16+1, 28+1, 8+1),
# # 	(32+1, 56+1, 16+1),
# # 	(64+1, 112+1, 32+1),
# # 	(128+1, 224+1, 64+1),
# # ],
# [
# 	(10+1, 14+1, 8+1),
# 	(20+1, 28+1, 16+1),
# 	(40+1, 56+1, 32+1),
# 	(80+1, 112+1, 64+1),
# 	(160+1, 224+1, 128+1),
# ],
# 'medium':
# [
# 	(14+1, 20+1, 8+1),
# 	(28+1, 40+1, 16+1),
# 	(56+1, 80+1, 32+1),
# 	(112+1, 160+1, 64+1),
# 	(224+1, 320+1, 128+1),
# ],
# 'fine':
# [
# 	(18+1, 24+1, 12+1),
# 	(36+1, 48+1, 24+1),
# 	(72+1, 96+1, 48+1),
# 	(144+1, 192+1, 96+1),
# 	(288+1, 384+1, 192+1),
# ]}
#point render
resolutions={'coarse':
[
	(18+1, 24+1, 12+1),
	(36+1, 48+1, 24+1),
	(72+1, 96+1, 48+1),
	(144+1, 192+1, 96+1),
	(288+1, 384+1, 192+1),
],
'medium':
[
	(20+1, 26+1, 14+1),
	(40+1, 52+1, 28+1),
	(80+1, 104+1, 56+1),
	(160+1, 208+1, 112+1),
	(320+1, 416+1, 224+1),
],
'fine':
[
	(20+1, 26+1, 14+1),
	(40+1, 52+1, 28+1),
	(80+1, 104+1, 56+1),
	(160+1, 208+1, 112+1),
	(320+1, 416+1, 224+1),
]
# 'fine':
# [
# 	(24+1, 30+1, 18+1),
# 	(48+1, 60+1, 36+1),
# 	(96+1, 120+1, 72+1),
# 	(192+1, 240+1, 144+1),
# 	(384+1, 480+1, 288+1),
# ]
}

resolutions_higher = [
	(32+1, 32+1, 32+1),
	(64+1, 64+1, 64+1),
	(128+1, 128+1, 128+1),
	(256+1, 256+1, 256+1),
	(512+1, 512+1, 512+1),
]



config=ConfigFactory.parse_file(args.conf)
if len(args.gpu_ids):
	device=torch.device(args.gpu_ids[0])
else:
	device=torch.device(0)
data_root=args.data
if args.save_folder is None:
	print('please set save-folder...')
	assert(False)

save_root=osp.join(data_root,args.save_folder)
debug_root=osp.join(save_root,'debug')
if not osp.isdir(save_root):
	os.makedirs(save_root)
if not osp.isdir(debug_root):
	os.makedirs(debug_root)
# save the config file
with open(osp.join(save_root,'config.conf'),'w') as ff:
	ff.write(HOCONConverter.convert(config,'hocon'))
if config.get_bool('train.idr_like_cond'):
	renderer_condlen={}
else:
	renderer_condlen={'renderer':config.get_int('render_net.condlen')}
batch_size=config.get_int('train.coarse.batch_size')
dataset,dataloader=getRigidDatasetAndLoader(data_root,renderer_condlen,batch_size,
						config.get_bool('train.shuffle'),config.get_int('train.num_workers'),
						config.get_bool('train.opt_R'),config.get_bool('train.opt_trans'),config.get_config('train.opt_camera'))


optNet,sdf_initialized=getRigidOptNet(dataset,batch_size,resolutions['coarse'],device,config,True)
optNet,dataloader=utils.set_hierarchical_config(config,'coarse',optNet,dataloader,resolutions['coarse'])


# saved_model_state = torch.load('Data/female-3-casual/result2/latest.pth',map_location=device)
# optNet.load_state_dict(saved_model_state["model_state_dict"])
# dataset.conds=[saved_model_state['dcond'].requires_grad_(),saved_model_state['rcond'].requires_grad_()]
if args.model is not None and osp.isfile(args.model):
	print('load model: '+args.model,end='')
	if args.sdf_model is not None:
		print(' and substitute sdf model with: '+args.sdf_model,end='')
		sdf_initialized=-1
	print()
	optNet,dataset=utils.load_rigid_model(args.model,optNet,dataset,device,config.get_bool('train.idr_like_cond'),args.sdf_model)

print('box:')
print(optNet.engine.b_min.view(-1).tolist())
print(optNet.engine.b_max.view(-1).tolist())
optNet.train()



learnable_ws=dataset.learnable_weights()


# cameras=optNet.maskRender.rasterizer.cameras

optimizer = torch.optim.Adam([{'params':learnable_ws},{'params':[p for p in optNet.parameters() if p.requires_grad]}], lr=config.get_float('train.learning_rate'))
# to do: change scheduler to be different for different param group
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.get_list('train.scheduler.milestones'), gamma=config.get_float('train.scheduler.factor'))

ratio={'sdfRatio':None,'renderRatio':None}
opt_times=0.
nepochs=config.get_int('train.nepoch')
sample_pix_num=config.get_int('train.sample_pix_num')
for epoch in range(0,nepochs+1):	
	if config.get_int('train.medium.start_epoch')>=0 and epoch==config.get_int('train.medium.start_epoch'):
		optNet,dataloader=utils.set_hierarchical_config(config,'medium',optNet,dataloader,resolutions['medium'])
		print('enable medium hierarchical')
		utils.save_rigid_model(osp.join(save_root,"coarse.pth"),epoch,optNet,dataset,config.get_bool('train.idr_like_cond'))
	if config.get_int('train.fine.start_epoch')>=0 and epoch==config.get_int('train.fine.start_epoch'):
		optNet,dataloader=utils.set_hierarchical_config(config,'fine',optNet,dataloader,resolutions['fine'])
		print('enable fine hierarchical')
		utils.save_rigid_model(osp.join(save_root,"medium.pth"),epoch,optNet,dataset,config.get_bool('train.idr_like_cond'))
	for data_index, (frame_ids, imgs, masks, albedos) in enumerate(dataloader):
		# print(frame_ids)
		# focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)
		# optNet.maskRender.rasterizer.cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
		frame_ids=frame_ids.long().to(device)
		imgs=imgs.to(device)
		masks=masks.to(device)
		albedos=albedos.to(device)
		optimizer.zero_grad()

		# ratio=float(float(epoch)*len(dataloader)+data_index)/(len(dataloader)*250)
		ratio['sdfRatio']=1.
		# ratio['renderRatio']=opt_times/(len(dataset)/4.*25)
		ratio['renderRatio']=1.
		# optNet.draw=True
		loss=optNet(imgs,masks,albedos,sample_pix_num,ratio,frame_ids,debug_root)		
		# optimizer.zero_grad() # reserve deformer weights update in computetmpps or here
		# assert(False)	
		loss.backward()	
		
		#rebuild the computation graph
		# poses,trans,d_cond,_=dataset.get_grad_parameters(frame_ids,device)
		# focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(batch_size,device)
		# optNet.maskRender.rasterizer.cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
		# cameras.focal_length=focals
		# cameras.principal_point=princeple_ps
		# cameras.R=Rs
		# cameras.T=Ts
		optNet.propagateTmpPsGrad(frame_ids,ratio)
		# for para in optNet.netRender.albedo.parameters():
		# 	para.grad.zero_()
		# for para in optNet.netRender.light.parameters():
		# 	para.grad.zero_()
		optimizer.step()
		if data_index%1==0:
			outinfo='(%d/%d): loss = %.5f; color_loss: %.5f, eikonal_loss: %.5f'%(epoch,data_index,loss.item(),optNet.info['color_loss'],optNet.info['grad_loss'])+ \
					(' albedo_loss: %.5f,'%optNet.info['albedo_loss'] if 'albedo_loss' in optNet.info else '')+ \
					(' shading_smooth: %.5f,'%optNet.info['shading_smooth'] if 'shading_smooth' in optNet.info else '')+ \
					(' shading_norm: %.5f,'%optNet.info['shading_norm'] if 'shading_norm' in optNet.info else '')
			outinfo+='\n'
			if 'mesh_loss' in optNet.info:
				outinfo+='\tmesh_sdf_l: %.5f, mesh_norm_l:%.5f; '%(optNet.info['mesh_loss_sdf'],optNet.info['mesh_loss_grad'])
				for k,v in optNet.info['mesh_loss'].items():
					outinfo+=k+': %.5f\t'%v
			else:
				outinfo+='\tpc_sdf_l: %.5f'%(optNet.info['pc_loss_sdf'])
				outinfo+=';\tpc_norm_l: %.5f; '%(optNet.info['pc_loss_norm']) if 'pc_loss_norm' in optNet.info else '; '
				for k,v in optNet.info['pc_loss'].items():
					outinfo+=k+': %.5f\t'%v
			outinfo+='\n\trayInfo(%d,%d)\tinvInfo(%d,%d)\tratio: (%.2f,%.2f)\tremesh: %.3f'%(*optNet.info['rayInfo'],*optNet.info['invInfo'],ratio['sdfRatio'],ratio['renderRatio'],optNet.info['remesh'])
			print(outinfo)

		opt_times+=1.
		# if data_index==2:
		# 	break
		# assert(False)

	optNet.draw=True
	utils.save_rigid_model(osp.join(save_root,"latest.pth"),epoch,optNet,dataset,config.get_bool('train.idr_like_cond'))
	scheduler.step()