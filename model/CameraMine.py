import numpy as np
import torch
from typing import Optional, Sequence, Tuple
from pytorch3d.renderer.cameras import (
	CamerasBase,
)
from pytorch3d.transforms import (
	Transform3d,
)
_R = torch.eye(3)[None]  # (1, 3, 3)
_T = torch.zeros(1, 3)  # (1, 3)
# in pytorch3d, the original PerspectiveCamera screen to ndc is a little different with Rasterizer
# if follow the pytorch interpolation description, in rasterizer, align_corners=False, in PerspectiveCamera, align_corners=True
# this class rectify PerspectiveCamera to keep consistent with Rasterizer
class RectifiedPerspectiveCameras(CamerasBase):
	"""
	A class which stores a batch of parameters to generate a batch of
	transformation matrices using the multi-view geometry convention for
	perspective camera.

	Parameters for this camera can be specified in NDC or in screen space.
	If you wish to provide parameters in screen space, you NEED to provide
	the image_size = (imwidth, imheight).
	If you wish to provide parameters in NDC space, you should NOT provide
	image_size. Providing valid image_size will triger a screen space to
	NDC space transformation in the camera.

	For example, here is how to define cameras on the two spaces.

	.. code-block:: python
		# camera defined in screen space
		cameras = PerspectiveCameras(
			focal_length=((22.0, 15.0),),  # (fx_screen, fy_screen)
			principal_point=((192.0, 128.0),),  # (px_screen, py_screen)
			image_size=((256, 256),),  # (imwidth, imheight)
		)

		# the equivalent camera defined in NDC space
		cameras = PerspectiveCameras(
			focal_length=((0.17875, 0.11718),),  # fx = fx_screen / half_imwidth,
												# fy = fy_screen / half_imheight
			principal_point=((-0.5, 0),),  # px = 1-1./imwidth-px_screen/half_imwidth,
										   # py = 1-1./imheight-py_screen/half_imheight
		)
	"""
	def __init__(self,focal_length=1.0,principal_point=((0.0, 0.0),),R=_R,T=_T,K=None,device="cpu",image_size=((-1,-1),),):
		super().__init__(device=device,focal_length=focal_length,principal_point=principal_point,R=R,T=T,K=K,image_size=image_size,)

	def get_projection_transform(self, **kwargs) -> Transform3d:
		K = kwargs.get("K", self.K)  # pyre-ignore[16]
		if K is not None:
			if K.shape != (self._N, 4, 4):
				msg = "Expected K to have shape of (%r, 4, 4)"
				raise ValueError(msg % (selimage_sizef._N))
		else:
			# pyre-ignore[16]
			image_size = kwargs.get("image_size", self.image_size)
			# if imwidth > 0, parameters are in screen space
			image_size = image_size if image_size[0][0] > 0 else None

			K = _get_sfm_calibration_matrix(
				self._N,
				self.device,
				kwargs.get("focal_length", self.focal_length),  # pyre-ignore[16]
				kwargs.get("principal_point", self.principal_point),  # pyre-ignore[16]
				orthographic=False,
				image_size=image_size,
			)

		transform = Transform3d(device=self.device)
		transform._matrix = K.transpose(1, 2).contiguous()
		return transform

	def unproject_points(self, xy_depth: torch.Tensor, world_coordinates: bool = True, **kwargs) -> torch.Tensor:
		if world_coordinates:
			to_ndc_transform = self.get_full_projection_transform(**kwargs)
		else:
			to_ndc_transform = self.get_projection_transform(**kwargs)

		unprojection_transform = to_ndc_transform.inverse()
		xy_inv_depth = torch.cat(
			(xy_depth[..., :2], 1.0 / xy_depth[..., 2:3]), dim=-1  # type: ignore
		)
		return unprojection_transform.transform_points(xy_inv_depth)

	#reload original to keep consistent with Rasterizer
	def transform_points_screen(self, points, image_size, eps: Optional[float] = None, **kwargs) -> torch.Tensor:
		"""
		Transform input points from world to screen space.

		Args:
			points: torch tensor of shape (N, V, 3).
			image_size: torch tensor of shape (N, 2)
			eps: If eps!=None, the argument is used to clamp the
				divisor in the homogeneous normalization of the points
				transformed to the ndc space. Please see
				`transforms.Transform3D.transform_points` for details.

				For `CamerasBase.transform_points`, setting `eps > 0`
				stabilizes gradients since it leads to avoiding division
				by excessivelly low numbers for points close to the
				camera plane.

		Returns
			new_points: transformed points with the same shape as the input.
		"""

		ndc_points = self.transform_points(points, eps=eps, **kwargs)

		if not torch.is_tensor(image_size):
			image_size = torch.tensor(
				image_size, dtype=torch.int64, device=points.device
			)
		if (image_size < 1).any():
			raise ValueError("Provided image size is invalid.")

		image_width, image_height = image_size.unbind(1)
		image_width = image_width.view(-1, 1)  # (N, 1)
		image_height = image_height.view(-1, 1)  # (N, 1)

		ndc_z = ndc_points[..., 2]
		screen_x = (image_width-1.)/2.-image_width*ndc_points[..., 0]/2.
		screen_y = (image_height-1.)/2.-image_height*ndc_points[..., 1]/2.

		return torch.stack((screen_x, screen_y, ndc_z), dim=2)

	# for my project, all batch camera share one screen space parameter, these parameter can be set as learnable
	# this function is used to compute rays 
	def view_rays(self,ps,cam_id=0):
		rays=torch.zeros_like(ps)
		rays[:,0]=-ps[:,0]/self.focal_length[cam_id,0]+ps[:,2]*self.principal_point[cam_id,0]/self.focal_length[cam_id,0]
		rays[:,1]=-ps[:,1]/self.focal_length[cam_id,1]+ps[:,2]*self.principal_point[cam_id,1]/self.focal_length[cam_id,1]
		rays[:,2]=ps[:,2]
		rays=rays/torch.norm(rays,p=2,dim=1,keepdim=True)
		rays=rays.matmul(self.R[cam_id].transpose(0,1))
		return rays

	def project(self,ps,cam_id=0):
		ps=ps.matmul(self.R[cam_id])+self.T[cam_id].view(1,3)
		x=self.principal_point[cam_id,0]-ps[:,0]*self.focal_length[cam_id,0]/ps[:,2]
		y=self.principal_point[cam_id,1]-ps[:,1]*self.focal_length[cam_id,1]/ps[:,2]
		return torch.cat([x.view(-1,1),y.view(-1,1)],dim=1)


	def angThreshold(self,pixoffset=0.4,cam_id=0):
		H=self.image_size[cam_id,1].item()
		W=self.image_size[cam_id,0].item()
		cx=self.principal_point[cam_id,0].item()
		cy=self.principal_point[cam_id,1].item()
		fx=self.focal_length[cam_id,0].item()
		fy=self.focal_length[cam_id,1].item()
		r1=torch.tensor([(W-cx)/fx,0.,1.])
		r2=torch.tensor([(W+pixoffset-cx)/fx,0.,1.])
		thred=torch.arcsin(r1.cross(r2).norm()/(r1.norm()*r2.norm()))/np.pi*180.

		r1=torch.tensor([-cx/fx,0.,1.])
		r2=torch.tensor([(pixoffset-cx)/fx,0.,1.])
		thred=torch.min(thred,torch.arcsin(r1.cross(r2).norm()/(r1.norm()*r2.norm()))/np.pi*180.)

		r1=torch.tensor([0.,(H-cy)/fy,1.])
		r2=torch.tensor([0.,(H+pixoffset-cy)/fy,1.])
		thred=torch.min(thred,torch.arcsin(r1.cross(r2).norm()/(r1.norm()*r2.norm()))/np.pi*180.)

		r1=torch.tensor([0.,-cy/fy,1.])
		r2=torch.tensor([0.,(pixoffset-cy)/fy,1.])
		thred=torch.min(thred,torch.arcsin(r1.cross(r2).norm()/(r1.norm()*r2.norm()))/np.pi*180.)
		return thred.item()

	def cam_pos(self,cam_id=0):
		return -self.R[cam_id].matmul(self.T[cam_id].view(-1,1)).view(-1)

def _get_sfm_calibration_matrix(N,
	device,
	focal_length,
	principal_point,
	orthographic: bool = False,
	image_size=None,
) -> torch.Tensor:
	"""
	Returns a calibration matrix of a perspective/orthograpic camera.

	Args:
		N: Number of cameras.
		focal_length: Focal length of the camera in world units.
		principal_point: xy coordinates of the center of
			the principal point of the camera in pixels.
		orthographic: Boolean specifying if the camera is orthographic or not
		image_size: (Optional) Specifying the image_size = (imwidth, imheight).
			If not None, the camera parameters are assumed to be in screen space
			and are transformed to NDC space.

		The calibration matrix `K` is set up as follows:

		.. code-block:: python

			fx = focal_length[:,0]
			fy = focal_length[:,1]
			px = principal_point[:,0]
			py = principal_point[:,1]

			for orthographic==True:
				K = [
						[fx,   0,    0,  px],
						[0,   fy,    0,  py],
						[0,    0,    1,   0],
						[0,    0,    0,   1],
				]
			else:
				K = [
						[fx,   0,   px,   0],
						[0,   fy,   py,   0],
						[0,    0,    0,   1],
						[0,    0,    1,   0],
				]

	Returns:
		A calibration matrix `K` of the SfM-conventioned camera
		of shape (N, 4, 4).
	"""

	if not torch.is_tensor(focal_length):
		focal_length = torch.tensor(focal_length, device=device)

	if focal_length.ndim in (0, 1) or focal_length.shape[1] == 1:
		fx = fy = focal_length
	else:
		fx, fy = focal_length.unbind(1)

	if not torch.is_tensor(principal_point):
		principal_point = torch.tensor(principal_point, device=device)

	px, py = principal_point.unbind(1)

	if image_size is not None:
		if not torch.is_tensor(image_size):
			image_size = torch.tensor(image_size, device=device)
		imwidth, imheight = image_size.unbind(1)
		# make sure imwidth, imheight are valid (>0)
		if (imwidth < 1).any() or (imheight < 1).any():
			raise ValueError(
				"Camera parameters provided in screen space. Image width or height invalid."
			)
		half_imwidth = imwidth / 2.0
		half_imheight = imheight / 2.0
		fx = fx / half_imwidth
		fy = fy / half_imheight
		px = 1.-1./imwidth-px/half_imwidth
		py = 1.-1./imheight-py/half_imheight

	K = fx.new_zeros(N, 4, 4)
	K[:, 0, 0] = fx
	K[:, 1, 1] = fy
	if orthographic:
		K[:, 0, 3] = px
		K[:, 1, 3] = py
		K[:, 2, 2] = 1.0
		K[:, 3, 3] = 1.0
	else:
		K[:, 0, 2] = px
		K[:, 1, 2] = py
		K[:, 3, 2] = 1.0
		K[:, 2, 3] = 1.0

	return K

class PointsRendererWithFrags(torch.nn.Module):
	"""
	A class for rendering a batch of points. The class should
	be initialized with a rasterizer and compositor class which each have a forward
	function.
	"""

	def __init__(self, rasterizer, compositor):
		super().__init__()
		self.rasterizer = rasterizer
		self.compositor = compositor

	def to(self, device):
		# Manually move to device rasterizer as the cameras
		# within the class are not of type nn.Module
		self.rasterizer = self.rasterizer.to(device)
		self.compositor = self.compositor.to(device)
		return self

	def forward(self, point_clouds, **kwargs) -> torch.Tensor:
		fragments = self.rasterizer(point_clouds, **kwargs)

		# Construct weights based on the distance of a point to the true point.
		# However, this could be done differently: e.g. predicted as opposed
		# to a function of the weights.
		r = self.rasterizer.raster_settings.radius

		dists2 = fragments.dists.permute(0, 3, 1, 2)
		weights = 1 - dists2 / (r * r)
		images = self.compositor(
			fragments.idx.long().permute(0, 3, 1, 2),
			weights,
			point_clouds.features_packed().permute(1, 0),
			**kwargs,
		)

		# permute so image comes at the end
		images = images.permute(0, 2, 3, 1)

		return images,fragments