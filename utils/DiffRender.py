import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftSilhouetteShader,
    BlendParams
)
import numpy as np
def InitPerspCameras(batchSize,focal_length,principal_point,image_size):
	return PerspectiveCameras(focal_length,principal_point,torch.eye(3)[None,:].repeat(batchSize,1,1),torch.zeros(batchSize,3),image_size=image_size)
def InitSoftSilhouette(image_size,cameras):
	sigma = 1.0e-5
	raster_settings_silhouette = RasterizationSettings(
    image_size=image_size, 
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
    # bin_size=0,
    faces_per_pixel=50,
    perspective_correct=True,
    clip_barycentric_coords=False,
    cull_backfaces=False
)	
	renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings_silhouette
    ),
    shader=SoftSilhouetteShader(BlendParams(sigma=sigma))
)
	return renderer_silhouette


if __name__ == '__main__':
	import cv2
	from time import time
	device=torch.device(0)
	meshs=load_objs_as_meshes(['../slower.obj'],load_textures=False,device=device)
	N=8
	meshes=meshs.extend(N).to(device)

	cam_pos= [torch.from_numpy(np.array([2.*np.sin(float(ind)/float(N) * 2.*np.pi),0,2.*np.cos(float(ind)/float(N) * 2.*np.pi)])).to(torch.float) for ind in range(N)]
	cam_pos=torch.stack(cam_pos).to(device)
	Rs,Ts=look_at_view_transform(eye=cam_pos,device=device)
	# Rs.requires_grad=True
	# Ts.requires_grad=True
	cameras=FoVPerspectiveCameras(znear=0.5, zfar=4., R=Rs, T=Ts,device=device,aspect_ratio=512./300.)
	render=InitSoftSilhouette((300,512),cameras).to(device)
	torch.cuda.synchronize()
	start=time()
	times=1
	for ind in range(times):
		# frags=render.rasterizer(meshes)
		# images=(frags.pix_to_face>=0).any(-1).float()
		images=render(meshes)[...,3]
	torch.cuda.synchronize()
	end=time()
	print('time:%.4f'%((end-start)/float(times)))
	images=(images*255.).detach().cpu().numpy().astype(np.uint8)
	for ind,img in enumerate(images):
		cv2.imwrite('test_mask/%d.png'%ind,img)
	print('done')


