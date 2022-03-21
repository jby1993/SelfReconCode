import torch
def smooth_weights(weights,times=3):
	for _ in range(times):
		mean=(weights[:,:,2:,1:-1,1:-1]+weights[:,:,:-2,1:-1,1:-1]+\
			weights[:,:,1:-1,2:,1:-1]+weights[:,:,1:-1,:-2,1:-1]+\
			weights[:,:,1:-1,1:-1,2:]+weights[:,:,1:-1,1:-1,:-2])/6.0
		weights[:,:,1:-1,1:-1,1:-1]=(weights[:,:,1:-1,1:-1,1:-1]-mean)*0.7+mean
		sums=weights.sum(1,keepdim=True)
		weights=weights/sums
	weights[weights<5.e-3]=0.0
	return weights

def compute_lbswField(bmins,bmaxs,resolutions,smpl_verts,smpl_ws,align_corners=False,mean_neighbor=5,smooth_times=30):
	device=smpl_verts.device
	bmins=torch.tensor(bmins).float().to(device).view(1,-1)
	bmaxs=torch.tensor(bmaxs).float().to(device).view(1,-1)
	W,H,D=resolutions
	resolutions = torch.tensor(resolutions).float().to(device).view(-1)
	arrangeX = torch.linspace(0, W-1, W).long().to(device)
	arrangeY = torch.linspace(0, H-1, H).long().to(device)
	arrangeZ = torch.linspace(0, D-1, D).long().to(device)
	gridD, girdH, gridW = torch.meshgrid([arrangeZ, arrangeY, arrangeX])
	coords = torch.stack([gridW, girdH, gridD]) # [3, steps[0], steps[1], steps[2]]
	coords = coords.view(3, -1).t() # [N, 3]
	
	if align_corners:
		coords2D = coords.float() / (resolutions[None,:] - 1)
	else:
		step = 1.0 / resolutions[None,:].float()
		coords2D = coords.float() / resolutions[None,:] + step / 2
	coords2D = coords2D * (bmaxs - bmins) + bmins
	fws=[]
	for ind,tmp in enumerate(torch.split(coords2D,50000)):
		# if ind/10==0:
		# 	print(ind)
		dists,indices=(tmp[:,None,:]-smpl_verts[None,:,:]).norm(dim=-1).topk(mean_neighbor,dim=-1,largest=False)
		dists=torch.clamp(dists,0.0001,1.)
		ws=1./dists
		ws=ws/ws.sum(-1,keepdim=True)
		# print(dists.shape)
		# print(indices.shape)
		# print(smpl_ws.shape)
		# print(ws.shape)
		ws=(smpl_ws[indices.view(-1)]*ws.view(-1,1)).reshape(ws.shape[0],mean_neighbor,-1).sum(1)
		fws.append(ws)

	fws=torch.cat(fws,dim=0)

	fws=fws.transpose(0,1).reshape(1,-1,D,H,W)
	fws=smooth_weights(fws,smooth_times)
	return fws

if __name__ == '__main__':
	from smpl_pytorch.SMPL import SMPL, getSMPL
	import cv2
	import numpy as np
	import os
	device=torch.device(2)
	smpl=getSMPL().to(device)
	if os.path.isfile('test_ws/ws.pth'):
		ws=torch.load('test_ws/ws.pth').to(device)
	else:		
		ws=compute_lbswField([-0.8,-1.2,-0.4],[0.8,0.8,0.4],(128+1, 320+1, 128+1),smpl.v_template,smpl.weight.view(6890,24),align_corners=False,mean_neighbor=30)
		torch.save(ws.detach().cpu(),'test_ws/ws.pth')
	colors=[(2,63,165),(125,135,185),(190,193,212),(214,188,192),(187,119,132),(142,6,59),(74,111,227),(133,149,225),(181,187,227),(230,175,185),(224,123,145),(211,63,106),(17,198,56),(141,213,147),(198,222,199),(234,211,198),(240,185,141),(239,151,8),(15,207,192),(156,222,214),(213,234,231),(243,225,235),(246,196,225),(247,156,212)]
	colors=np.array(colors,dtype=np.float32)/255.
	colors=torch.from_numpy(colors[np.random.permutation(colors.shape[0])]).to(device)
	W=801
	H=1001
	arrangeX = torch.linspace(0, W-1, W).long().to(device)
	arrangeY = torch.linspace(0, H-1, H).long().to(device)
	gridH, gridW = torch.meshgrid([arrangeY, arrangeX])
	gridW=2.*gridW.float()/W+1./W - 1.
	gridH=2.*gridH.float()/H+1./H - 1.
	grid = torch.stack([gridW, gridH, torch.zeros_like(gridW)]).permute(1,2,0).unsqueeze(0).unsqueeze(0)
	outs=torch.nn.functional.grid_sample(ws, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0,:,0,:,:].permute(1,2,0)
	outs=outs.reshape(W*H,-1).matmul(colors)
	outs=torch.clamp(outs.reshape(H,W,-1),0,1.)*255.
	img=outs.detach().cpu().numpy().astype(np.uint8)[::-1,:,:]
	cv2.imwrite('test_ws/mixed.png',img)

	grid = torch.stack([gridW, gridH, 0.2*torch.ones_like(gridW)]).permute(1,2,0).unsqueeze(0).unsqueeze(0)
	outs=torch.nn.functional.grid_sample(ws, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0,:,0,:,:].permute(1,2,0)
	outs=outs.reshape(W*H,-1).matmul(colors)
	outs=torch.clamp(outs.reshape(H,W,-1),0,1.)*255.
	img=outs.detach().cpu().numpy().astype(np.uint8)[::-1,:,:]
	cv2.imwrite('test_ws/mixedf.png',img)

	grid = torch.stack([gridW, gridH, -0.2*torch.ones_like(gridW)]).permute(1,2,0).unsqueeze(0).unsqueeze(0)
	outs=torch.nn.functional.grid_sample(ws, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0,:,0,:,:].permute(1,2,0)
	outs=outs.reshape(W*H,-1).matmul(colors)
	outs=torch.clamp(outs.reshape(H,W,-1),0,1.)*255.
	img=outs.detach().cpu().numpy().astype(np.uint8)[::-1,:,:]
	cv2.imwrite('test_ws/mixedb.png',img)
	# for ind,img in enumerate(outs):
	# 	img=(img[:,:]*255.).detach().cpu().numpy().astype(np.uint8)
	# 	cv2.imwrite('test_ws/%d.png'%(ind),img)

	from pytorch3d.structures import Meshes
	from pytorch3d.renderer import (
	    FoVOrthographicCameras,
	    RasterizationSettings, 
	    MeshRasterizer
	)


	raster_settings_silhouette = RasterizationSettings(
	image_size=(H,W), 
	blur_radius=0, 
	faces_per_pixel=1,
	perspective_correct=True,
	clip_barycentric_coords=False,
	cull_backfaces=True
)	
	cameras=FoVOrthographicCameras(znear=1.6,zfar=2.4,max_y=0.8,min_y=-1.2,max_x=0.8,min_x=-0.8,R=torch.tensor([-1,1,-1]).diag().unsqueeze(0),T=torch.tensor([[0,0,2.]]))
	rasterizer=MeshRasterizer(
		cameras=cameras, 
		raster_settings=raster_settings_silhouette
	).to(device)
	faces=torch.from_numpy(np.array(smpl.faces,dtype=np.int64)).to(device)
	frags=rasterizer(Meshes(verts=smpl.v_template.view(1,6890,3),faces=faces.view(1,-1,3)).to(device))
	bary_coords=frags.bary_coords.reshape(H*W,3)
	pix_to_face=frags.pix_to_face.reshape(H*W,1)

	vcolors=smpl.weight.view(6890,24)@colors

	selindices=(pix_to_face>=0).view(-1)
	vcolors=vcolors[faces[pix_to_face[selindices]].view(-1)].view(-1,3,3)
	vcolors=torch.clamp((vcolors*bary_coords[selindices].view(-1,3,1)).sum(1),0.,1.)*255.
	img=np.zeros((H*W,3),dtype=np.uint8)
	img[selindices.cpu().numpy()]=vcolors.cpu().numpy().astype(np.uint8)
	cv2.imwrite('test_ws/smpl.png',img.reshape(H,W,3))
	print('done')