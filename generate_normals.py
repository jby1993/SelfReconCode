import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from lib.networks import define_G
from glob import glob
import argparse
import os
import os.path as osp
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
parser = argparse.ArgumentParser(description='neu video body rec')
parser.add_argument('--gid',default=0,type=int,metavar='ID',
					help='gpu id')
parser.add_argument('--imgpath',default=None,metavar='M',
					help='config file')
args = parser.parse_args()


def crop_image(img, rect):
	x, y, w, h = rect

	left = abs(x) if x < 0 else 0
	top = abs(y) if y < 0 else 0
	right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0
	bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0
	
	if img.shape[2] == 4:
		color = [0, 0, 0, 0]
	else:
		color = [0, 0, 0]
	new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

	x = x + left
	y = y + top

	return new_img[y:(y+h),x:(x+w),:]



class EvalDataset(Dataset):
	def __init__(self, root):
		self.root=root
		self.img_files=[osp.join(self.root,f) for f in os.listdir(self.root) if f.split('.')[-1] in ['png', 'jpeg', 'jpg', 'PNG', 'JPG', 'JPEG'] and osp.exists(osp.join(self.root,f.replace('.%s' % (f.split('.')[-1]), '_rect.txt')))]
		self.img_files.sort(key=lambda x: int(osp.basename(x).split('.')[0]))
		# self.img_files=sorted([osp.join(self.root,f) for f in ['0.png'] if f.split('.')[-1] in ['png', 'jpeg', 'jpg', 'PNG', 'JPG', 'JPEG'] and osp.exists(osp.join(self.root,f.replace('.%s' % (f.split('.')[-1]), '_rect.txt')))])

		self.to_tensor = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		])
		self.person_id=0
	def __len__(self):
		return len(self.img_files)
	def get_item(self, index):
		img_path = self.img_files[index]
		rect_path = self.img_files[index].replace('.%s' % (self.img_files[index].split('.')[-1]), '_rect.txt')
		mask_path=self.img_files[index].replace('/imgs/','/masks/')[:-3]+'png'

		# Name
		img_name = os.path.splitext(os.path.basename(img_path))[0]

		im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
		# print(mask_path)
		if osp.isfile(mask_path):
			mask=cv2.imread(mask_path)
			bg=~(mask>0).all(-1)
			im[bg]=np.zeros(im.shape[-1],dtype=im.dtype)
		else:
			bg=None
		H,W=im.shape[:2]
		if im.shape[2] == 4:
			im = im / 255.0
			im[:,:,:3] /= im[:,:,3:] + 1e-8
			im = im[:,:,3:] * im[:,:,:3] + 0.5 * (1.0 - im[:,:,3:])
			im = (255.0 * im).astype(np.uint8)
		h, w = im.shape[:2]
		

		rects = np.loadtxt(rect_path, dtype=np.int32)
		if len(rects.shape) == 1:
			rects = rects[None]
			pid=0
		else:
			max_len=0
			pid=-1
			for ind,rect in enumerate(rects):
				cur_len=(rect[-2]+rect[-1])//2
				if max_len<cur_len:
					max_len=cur_len
					pid=ind
		# pid = min(rects.shape[0]-1, self.person_id)

		rect = rects[pid].tolist()
		im = crop_image(im, rect)
		im_512 = cv2.resize(im, (512, 512))
		image_512 = Image.fromarray(im_512[:,:,::-1]).convert('RGB')

		# image
		image_512 = self.to_tensor(image_512)
		return (img_name,image_512.unsqueeze(0),bg,H,W,rect)

	def __getitem__(self, index):
		return self.get_item(index)




device=torch.device(args.gid)


# save_root=osp.normpath(osp.join(args.imgpath,osp.pardir,'normals'))
# os.makedirs(save_root,exist_ok=True)

netF=define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

weights={}
for k,v in torch.load('checkpoints/pifuhd.pt',map_location='cpu')['model_state_dict'].items():
	if k[:10]=='netG.netF.':
		weights[k[10:]]=v

netF.load_state_dict(weights)

netF=netF.to(device)

netF.eval()
cids=[temp for temp in os.listdir(args.imgpath) if osp.isdir(osp.join(args.imgpath,temp)) and temp.isdigit()]
if len(cids)==0:
	cids=['.']
for fold in cids:
	save_root=osp.normpath(osp.join(args.imgpath,osp.pardir,'normals',fold))
	print(save_root)
	os.makedirs(save_root,exist_ok=True)
	dataset=EvalDataset(osp.normpath(osp.join(args.imgpath,fold)))
	writer=None
	with torch.no_grad():	
		for i in tqdm(range(len(dataset))):
			img_name,img,bg,H,W,rect=dataset[i]
			if writer is None:
				writer=cv2.VideoWriter(osp.join(save_root,'video.avi'),cv2.VideoWriter.fourcc('M','J','P','G'),30.,(W,H))		
			x,y,w,h=[float(tmp) for tmp in rect]
			# cv2.imwrite('test.png',((np.transpose(img.numpy()[0],(1,2,0))*0.5+0.5)[:,:,::-1]*255.0).astype(np.uint8))
			
			img=img.to(device)

			nml=netF.forward(img)

			gridH,gridW=torch.meshgrid([torch.arange(H).float().to(device),torch.arange(W).float().to(device)])
			coords=torch.stack([gridW,gridH]).permute(1,2,0).unsqueeze(0)
			coords[...,0]=2.0*(coords[...,0]-x)/w-1.0
			coords[...,1]=2.0*(coords[...,1]-y)/h-1.0
			nml=torch.nn.functional.grid_sample(nml,coords,mode='bilinear', padding_mode='zeros', align_corners=True)

			unvalid_mask=(torch.norm(nml,dim=1)<0.0001).detach().cpu().numpy()[0]
			nml=nml.detach().cpu().numpy()[0]
			nml=(np.transpose(nml,(1,2,0))*0.5+0.5)[:,:,::-1]*255.0
			if unvalid_mask.sum()>0:
				nml[unvalid_mask]=0.
			# print(osp.join(save_root,img_name,'.png'))
			if bg is not None:
				nml[bg]=0.
			# if (unvalid_mask*(~bg)).sum()>0:
			# 	print(i)
			cv2.imwrite(osp.join(save_root,img_name+'.png'),nml.astype(np.uint8))
			writer.write(nml.astype(np.uint8))

	if writer is not None:
		writer.release()
print('done.')