# import text_to_image
import os
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from textwrap3 import wrap
import ctypes

def read_text_from_text_file_path(text_file_path):
	f = open(text_file_path, "r")
	return(f.read())

def draw_image(text,fullpath,width,height):

    # width=500
    # height=1000
    back_ground_color=(255,255,255)
    font_size=10
    font_color=(0,0,0)
    unicode_text = "\n".join(wrap(text,width))

    im  =  Image.new ( "RGB", (width,height), back_ground_color )
    draw  =  ImageDraw.Draw (im)
    unicode_font = ImageFont.truetype("arial.ttf", font_size)
    draw.text ( (10,10), unicode_text, font=unicode_font, fill=font_color )
    im.save(fullpath)
    return fullpath

def argCheck_convert_text_to_image(text_file_path,image_dataset_path,index):
	if type(index)!=int:
		return f"{index} is not integer!"
	if type(text_file_path)!=str:
		return f"{text_file_path} is not string!"
	if type(image_dataset_path)!=str:
		return f"{text_file_path} is not string!"
	if not os.path.exists(text_file_path):
		return f"{text_file_path} does not exists!"
	for image_path in image_dataset_path:
		if not os.path.exists(image_path):
			return f"{image_path} does not exists!"
	return True

def convert_text_to_image(text_file_path,image_dataset_path,index):
	check = argCheck_convert_text_to_image(text_file_path,image_dataset_path,index)
	if not check:return check
	user32 = ctypes.windll.user32
	screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
	text = read_text_from_text_file_path(text_file_path)

	fullpath = os.path.join(image_dataset_path,f"image-{index}.png")
	encoded_image_path = draw_image(text, fullpath,screensize[0],screensize[1])
	# encoded_image_path = text_to_image.encode_file(text_file_path, os.path.join(dataset_path,f"image-{index}.png"))
	return encoded_image_path


def argCheck_image_to_text_engineering(image_dataset_path,image_to_text_dict,Type):
	if type(image_dataset_path)!=str:
		return f"{text_file_path} is not string!"
	if type(image_to_text_dict)!=dict:
		return f"{text_file_path} is not dictionary!"
	if type(Type)!=str:
		return f"{Type} is not string!"
	for image_path in image_dataset_path:
		if not os.path.exists(image_path):
			return f"{image_path} does not exists!"
	for image,_ in image_to_text_dict.items():
		try:
			im = Image.open(image)
		except Exception as e:
			return str(e)
	if Type not in ["PHash","AHash","WHash","CNN"]:
		return f"{Type} not one of PHash, AHash, WHash and CNN!"
	return True


def image_to_text_engineering(image_dataset_path,image_to_text_dict,type = "PHash"):
	check = argCheck_image_to_text_engineering(image_dataset_path,image_to_text_dict,type)
	if not check:return check
	from imagededup.methods import PHash,DHash,WHash,AHash,CNN
	hasher = eval(f"{type}()")
	encodings = hasher.encode_images(image_dir=image_dataset_path)
	duplicates = hasher.find_duplicates(encoding_map=encodings)
	# pprint(duplicates)
	duplicates_text = {}
	for k,v in duplicates.items():
		x = image_to_text_dict[os.path.join(image_dataset_path,k)]
		y = [image_to_text_dict[os.path.join(image_dataset_path,vv)] for vv in v]
		duplicates_text[x] = y 
	# pprint(duplicates_text)
	return duplicates_text

if __name__=="__main__":
	text_files = [
		r"C:\Users\gprak\Downloads\text_dataset\text1.txt",
		r"C:\Users\gprak\Downloads\text_dataset\text2.txt",
		r"C:\Users\gprak\Downloads\text_dataset\text3.txt"
	]
	image_dataset_path = r"C:\Users\gprak\Downloads\image_dataset"
	encoded_images_path = [convert_text_to_image(text_file_path,image_dataset_path,k) for k,text_file_path in enumerate(text_files)]
	image_to_text_dict = {}
	for encoded_image_path,text_file in zip(encoded_images_path,text_files):
		image_to_text_dict[encoded_image_path] = text_file

	from pprint import pprint
	pprint(encoded_images_path)
	
	duplicates_text = image_to_text_engineering(image_dataset_path,image_to_text_dict,"CNN")
	pprint(duplicates_text)