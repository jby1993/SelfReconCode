#use VideoAvatar environment
import os
import os.path as osp
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
from opendr.geometry import VertNormals
from tex.iso import Isomapper, IsoColoredRenderer
from glob import glob
import cv2
import argparse
parser = argparse.ArgumentParser(description='neu video body infer')
parser.add_argument('--tmp-root',default=None,metavar='M',
					help='data root')
parser.add_argument('--res',default=1680,type=int,metavar='IDs',
					help='texture resolution')
args = parser.parse_args()

# rec_root='/data/jby/NeuVideoRec/Data/NeuralBody/IMG_1440/new_config_136/'
root=osp.normpath(osp.join(args.tmp_root,osp.pardir))
assert(osp.isfile(osp.join(args.tmp_root,'tex_predata.npz')))
resolution=args.res


data=np.load(osp.join(args.tmp_root,'tex_predata.npz'))
indices_texture = data['fids']
mask_root=osp.normpath(osp.join(root,osp.pardir,'masks'))
img_root=osp.normpath(osp.join(root,osp.pardir,'imgs'))

# print(indices_texture.shape)

img_ns=glob(osp.join(img_root,'*.jpg'))
img_ns.extend(glob(osp.join(img_root,'*.png')))
img_ns.sort(key=lambda x: int(osp.basename(x).split('.')[0]))
mask_ns=glob(osp.join(mask_root,'*.png'))
mask_ns.sort(key=lambda x: int(osp.basename(x).split('.')[0]))


vt=data['vt']
ft=data['ft']
defVs=data['defVs']

# print(vt.shape, ft.shape, defVs.shape, data['tmpvs'].shape)
# print(data['fs'].shape)


bgcolor = np.array([1., 0.2, 1.])
iso = Isomapper(vt, ft, data['fs'], resolution, bgcolor=bgcolor)
iso_vis = IsoColoredRenderer(vt, ft, data['fs'], resolution)
camera = ProjectPoints(t=data['cam_t'], rt=data['cam_rt'], c=data['cam_c'],
					   f=data['cam_f'], k=data['cam_k'], v=data['tmpvs'])
img=cv2.imread(img_ns[0])
frustum = {'near': 0.1, 'far': 1000., 'width': img.shape[1], 'height': img.shape[0]}
rn_vis = ColoredRenderer(f=data['fs'], frustum=frustum, camera=camera, num_channels=1)


agg_num=50
normal_ang=68
check_num=5


tex_agg = np.zeros((resolution, resolution, agg_num, 3))
tex_agg[:] = np.nan
viewid_agg = -np.ones((resolution, resolution, agg_num),dtype=np.int)
normal_initial=np.cos(normal_ang/180.*np.pi)
normal_agg = np.ones((resolution, resolution, agg_num)) *normal_initial

vn = VertNormals(f=data['fs'], v=data['tmpvs'])
static_indices = np.indices((resolution, resolution))

for ind,i in enumerate(indices_texture):
	print('Getting part texture from frame {}...'.format(i))
	# print(img_ns[i])
	# print(int(osp.basename(img_ns[i]).split('.')[0]))
	assert(osp.isfile(img_ns[i]) and int(osp.basename(img_ns[i]).split('.')[0])==i)
	assert(osp.isfile(mask_ns[i]) and int(osp.basename(mask_ns[i]).split('.')[0])==i)
	frame=cv2.imread(img_ns[i])
	mask=cv2.imread(mask_ns[i]).astype(np.uint8)
	if len(mask.shape)>2:
		mask=np.any(mask,axis=-1)

	camera.v=defVs[ind]
	vn.v=defVs[ind]

	visibility = rn_vis.visibility_image.ravel()
	visible = np.nonzero(visibility != 4294967295)[0]

	proj = camera.r
	in_viewport = np.logical_and(
		np.logical_and(np.round(camera.r[:, 0]) >= 0, np.round(camera.r[:, 0]) < frustum['width']),
		np.logical_and(np.round(camera.r[:, 1]) >= 0, np.round(camera.r[:, 1]) < frustum['height']),
	)
	in_mask = np.zeros(camera.shape[0], dtype=np.bool)
	idx = np.round(proj[in_viewport][:, [1, 0]].T).astype(np.int).tolist()
	in_mask[in_viewport] = mask[idx]

	faces_in_mask = np.where(np.min(in_mask[data['fs']], axis=1))[0]
	visible_faces = np.intersect1d(faces_in_mask, visibility[visible])

	# get the current unwrap
	part_tex = iso.render(frame / 255., camera, visible_faces)

	# angle under which the texels have been seen
	points = np.hstack((proj, np.ones((proj.shape[0], 1))))
	points3d = camera.unproject_points(points)
	points3d /= np.linalg.norm(points3d, axis=1).reshape(-1, 1)
	alpha = np.sum(points3d * -vn.r, axis=1).reshape(-1, 1)
	alpha[alpha < 0] = 0
	iso_normals = iso_vis.render(alpha)[:, :, 0]
	iso_normals[np.all(part_tex == bgcolor, axis=2)] = 0

	# texels to consider
	part_mask = np.zeros((resolution, resolution))
	min_normal = np.min(normal_agg, axis=2)
	part_mask[iso_normals > min_normal] = 1.

	# update best seen texels
	where = np.argmax(np.atleast_3d(iso_normals) - normal_agg, axis=2)

	idx = np.dstack((static_indices[0], static_indices[1], where))[part_mask == 1]
	tex_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = part_tex[part_mask == 1]
	viewid_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = i
	normal_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = iso_normals[part_mask == 1]



print('Inpainting unseen areas...')
# print(normal_agg.shape)
# where = np.max(normal_agg, axis=2) > normal_initial

viewid=viewid_agg[static_indices[0],static_indices[1],np.argmax(normal_agg,axis=2)]
where = np.sum(normal_agg>normal_initial,axis=2)>=check_num
viewid[~where]=-1
# cv2.imwrite(osp.join(args.tmp_root,'view_id.png'),viewid.astype(np.uint16))
tex_mask = iso.iso_mask
cv2.imwrite(osp.join(args.tmp_root,'tex_mask.png'), np.uint8(tex_mask * 255))
mask_final = np.float32(where)
cv2.imwrite(osp.join(args.tmp_root,'mask_final.png'), np.uint8(mask_final * 255))
# merge textures
print('Computing median texture...')
# print(tex_agg.shape)
tex_median = np.nanmedian(tex_agg, axis=2)
tex_median[np.all(np.atleast_3d(mask_final)<0.5,axis=2),:]=np.zeros((1,3))
cv2.imwrite(osp.join(args.tmp_root,'tex_median.png'), np.uint8(tex_median * 255))


kernel_size = np.int(resolution * 0.1)
kernel = np.ones((kernel_size, kernel_size), np.uint8)
inpaint_area = cv2.dilate(tex_mask, kernel) - mask_final
# cv2.imwrite(osp.join(args.tmp_root,'inpaint_area.png'), np.uint8(inpaint_area * 255))
tex_final = cv2.inpaint(np.uint8(tex_median * 255), np.uint8(inpaint_area * 255), 3, cv2.INPAINT_TELEA)

cv2.imwrite(osp.join(args.tmp_root,'texture.png'), tex_final)

# #debug
# np.savez(osp.join(args.tmp_root,'debug.npz'),tex_median=tex_median,where=where,tex_mask=tex_mask,resolution=resolution)
print('Done.')