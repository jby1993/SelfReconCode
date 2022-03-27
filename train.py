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
parser = argparse.ArgumentParser(description='neu video body rec')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
					help='gpu ids')
parser.add_argument('--conf',default=None,metavar='M',
					help='config file')
parser.add_argument('--data',default=None,metavar='M',
					help='data root')
parser.add_argument('--model',default=None,metavar='M',
					help='pretrained scene model')
parser.add_argument('--model-rm-prefix',nargs='+',type=str,metavar='rm prefix', help='rm model prefix')
parser.add_argument('--sdf-model',default=None,metavar='M',
					help='substitute sdf model')
parser.add_argument('--save-folder',default=None,metavar='M',help='save folder')
args = parser.parse_args()


#point render
resolutions={'coarse':
[
	(14+1, 20+1, 8+1),
	(28+1, 40+1, 16+1),
	(56+1, 80+1, 32+1),
	(112+1, 160+1, 64+1),
	(224+1, 320+1, 128+1),
],
'medium':
[
	(18+1, 24+1, 12+1),
	(36+1, 48+1, 24+1),
	(72+1, 96+1, 48+1),
	(144+1, 192+1, 96+1),
	(288+1, 384+1, 192+1),
],
'fine':
[
	(20+1, 26+1, 14+1),
	(40+1, 52+1, 28+1),
	(80+1, 104+1, 56+1),
	(160+1, 208+1, 112+1),
	(320+1, 416+1, 224+1),
]
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
os.makedirs(save_root,exist_ok=True)
os.makedirs(debug_root,exist_ok=True)
# save the config file
with open(osp.join(save_root,'config.conf'),'w') as ff:
	ff.write(HOCONConverter.convert(config,'hocon'))
condlen={'deformer':config.get_int('mlp_deformer.condlen'),'renderer':config.get_int('render_net.condlen')}
batch_size=config.get_int('train.coarse.point_render.batch_size')
dataset,dataloader=getDatasetAndLoader(data_root,condlen,batch_size,
						config.get_bool('train.shuffle'),config.get_int('train.num_workers'),
						config.get_bool('train.opt_pose'),config.get_bool('train.opt_trans'),config.get_config('train.opt_camera'))

# bmins=[-0.8,-1.25,-0.4]
# bmaxs=[0.8,0.7,0.4]
# use adaptive box computation
bmins=None
bmaxs=None

if config.get_int('train.initial_iters')<=0:
	use_initial_sdf=True
else:
	use_initial_sdf=False
optNet,sdf_initialized=getOptNet(dataset,batch_size,bmins,bmaxs,resolutions['coarse'],device,config,use_initial_sdf)
optNet,dataloader=utils.set_hierarchical_config(config,'coarse',optNet,dataloader,resolutions['coarse'])


if args.model is not None and osp.isfile(args.model):
	print('load model: '+args.model,end='')
	if args.sdf_model is not None:
		print(' and substitute sdf model with: '+args.sdf_model,end='')
		sdf_initialized=-1
	print()
	optNet,dataset=utils.load_model(args.model,optNet,dataset,device,args.sdf_model,args.model_rm_prefix)

print('box:')
print(optNet.engine.b_min.view(-1).tolist())
print(optNet.engine.b_max.view(-1).tolist())
optNet.train()

if sdf_initialized>0:
	optNet.initializeTmpSDF(sdf_initialized,osp.join(data_root,'initial_sdf_idr'+'_%d_%d.pth'%(config.get_int('sdf_net.multires'),config.get_int('train.skinner_pose_type'))),True)
	engine = Seg3dLossless(
			query_func=None, 
			b_min = optNet.engine.b_min,
			b_max = optNet.engine.b_max,
			resolutions=resolutions['coarse'],
			align_corners=False,
			balance_value=0.0, # be careful
			visualize=False,
			debug=False,
			use_cuda_impl=False,
			faster=False 
		).to(device)
	verts,faces=optNet.discretizeSDF(-1,engine)
	mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy())

	mesh.export(osp.join(data_root,'initial_sdf_idr'+'_%d_%d.ply'%(config.get_int('sdf_net.multires'),config.get_int('train.skinner_pose_type'))))


learnable_ws=dataset.learnable_weights()



optimizer = torch.optim.Adam([{'params':learnable_ws},{'params':[p for p in optNet.parameters() if p.requires_grad]}], lr=config.get_float('train.learning_rate'))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.get_list('train.scheduler.milestones'), gamma=config.get_float('train.scheduler.factor'))

ratio={'sdfRatio':None,'deformerRatio':None,'renderRatio':None}
opt_times=0.
nepochs=config.get_int('train.nepoch')
sample_pix_num=config.get_int('train.sample_pix_num')
in_fine_hie=False
for epoch in range(0,nepochs+1):	
	if config.get_int('train.medium.start_epoch')>=0 and epoch==config.get_int('train.medium.start_epoch'):
		optNet,dataloader=utils.set_hierarchical_config(config,'medium',optNet,dataloader,resolutions['medium'])
		torch.cuda.empty_cache()
		print('enable medium hierarchical')
		utils.save_model(osp.join(save_root,"coarse.pth"),epoch,optNet,dataset)
	if config.get_int('train.fine.start_epoch')>=0 and epoch==config.get_int('train.fine.start_epoch'):
		optNet,dataloader=utils.set_hierarchical_config(config,'fine',optNet,dataloader,resolutions['fine'])
		print('enable fine hierarchical')
		torch.cuda.empty_cache()
		utils.save_model(osp.join(save_root,"medium.pth"),epoch,optNet,dataset)
		in_fine_hie=True
	# for data_index, (frame_ids, imgs, masks, albedos) in enumerate(dataloader):
	for data_index, (frame_ids, outs) in enumerate(dataloader):
		frame_ids=frame_ids.long().to(device)
		optimizer.zero_grad()

		ratio['sdfRatio']=1.
		ratio['deformerRatio']=opt_times/2500.+0.5
		ratio['renderRatio']=1.
		loss=optNet(outs,sample_pix_num,ratio,frame_ids,debug_root)		
		loss.backward()	
		optNet.propagateTmpPsGrad(frame_ids,ratio)
		optimizer.step()
		if data_index%1==0:
			outinfo='(%d/%d): loss = %.5f; color_loss: %.5f, eikonal_loss: %.5f'%(epoch,data_index,loss.item(),optNet.info['color_loss'],optNet.info['grad_loss'])+ \
					(' normal_loss: %.5f,'%optNet.info['normal_loss'] if 'normal_loss' in optNet.info else '')+ \
					(' def_loss: %.5f,'%optNet.info['def_loss'] if 'def_loss' in optNet.info else '')+ \
					(' offset_loss: %.5f,'%optNet.info['offset_loss'] if 'offset_loss' in optNet.info else '')+ \
					(' dct_loss: %.5f,'%optNet.info['dct_loss'] if 'dct_loss' in optNet.info else '')
			outinfo+='\n'
			outinfo+='\tpc_sdf_l: %.5f'%(optNet.info['pc_loss_sdf'])
			outinfo+=';\tpc_norm_l: %.5f; '%(optNet.info['pc_loss_norm']) if 'pc_loss_norm' in optNet.info else '; '
			for k,v in optNet.info['pc_loss'].items():
				outinfo+=k+': %.5f\t'%v
			outinfo+='\n\trayInfo(%d,%d)\tinvInfo(%d,%d)\tratio: (%.2f,%.2f,%.2f)\tremesh: %.3f'%(*optNet.info['rayInfo'],*optNet.info['invInfo'],ratio['sdfRatio'],ratio['deformerRatio'],ratio['renderRatio'],optNet.info['remesh'])
			print(outinfo)

		opt_times+=1.
	if in_fine_hie:
		optNet.draw=True
	utils.save_model(osp.join(save_root,"latest.pth"),epoch,optNet,dataset)
	scheduler.step()