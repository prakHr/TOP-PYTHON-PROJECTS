from pelican.plugins import thumbnailer
from PIL import Image

def create_thumbnail(type,dim,path,img_path,save_path,want_to_save_path=False):
	img = Image.open(img_path)
	r = thumbnailer.Resizer(type,dim,path)
	output = r.resize(img)
	if want_to_save_path:
		if os.path.exists(save_path):
			output.save(save_path)
	return output