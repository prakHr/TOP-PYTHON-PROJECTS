from PIL import Image
import easyocr
import numpy as np
from ftfy import fix_encoding,fix_text
import random
from deep_translator import GoogleTranslator
from workbook import Workbook


def garbageCleaner(garbage_text):
    rv = fix_encoding(garbage_text)
    rv = fix_text(rv)
    dict = {"garbage_text":garbage_text,"cleaned_text":rv}
    return dict

def garbagesCleaner(garbage_texts):
    rv = [garbageCleaner(garbage_text) for garbage_text in garbage_texts]
    dv = {}
    for garbage_text in garbage_texts:
    	dv[garbage_text] = ""
    rv = list(dv.keys())
    return rv



def image_to_text_function(pil_image):
    "Extract Translated Text from Image"
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    open_cv_image = np.array(pil_image)
    result = reader.readtext(open_cv_image)
    textt = " ".join([b for a,b,c in result])
    garbage_text = textt
    ans = garbageCleaner(garbage_text)
    textt = ans["cleaned_text"]
    to_translate = textt
    available_languages = {'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'assamese': 'as', 'aymara': 'ay', 'azerbaijani': 'az', 'bambara': 'bm', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bhojpuri': 'bho', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-CN', 'chinese (traditional)': 'zh-TW', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dhivehi': 'dv', 'dogri': 'doi', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'ewe': 'ee', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'guarani': 'gn', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'ilocano': 'ilo', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'kinyarwanda': 'rw', 'konkani': 'gom', 'korean': 'ko', 'krio': 'kri', 'kurdish (kurmanji)': 'ku', 'kurdish (sorani)': 'ckb', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lingala': 'ln', 'lithuanian': 'lt', 'luganda': 'lg', 'luxembourgish': 'lb', 'macedonian': 'mk', 'maithili': 'mai', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'meiteilon (manipuri)': 'mni-Mtei', 'mizo': 'lus', 'mongolian': 'mn', 'myanmar': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia (oriya)': 'or', 'oromo': 'om', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'quechua': 'qu', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'sanskrit': 'sa', 'scots gaelic': 'gd', 'sepedi': 'nso', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'tatar': 'tt', 'telugu': 'te', 'thai': 'th', 'tigrinya': 'ti', 'tsonga': 'ts', 'turkish': 'tr', 'turkmen': 'tk', 'twi': 'ak', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}
   
    textt = ""
    rv = {}
    for _,language in available_languages.items():
        translated = GoogleTranslator(source='auto', target=language).translate(to_translate)
        textt += language+" = " + translated + "\n"
        rv[language] = translated
    return textt,rv



def automateExcelCreation(wb,data,sheet_name):
	wb.write_sheet(data,sheet_name,print_to_screen=False)
	return wb 


def automateMultipleExcelCreation(list_of_data,list_of_sheet_name):
	wb = Workbook(style_compression=2)
	for data,sheet_name in zip(list_of_data,list_of_sheet_name):
		wb = automateExcelCreation(wb,data,sheet_name)
	return wb

def argCheck_allProjects_combinedPipeline(image_path):
	import os
	if not os.path.exists(image_path):
		return "image path is not valid!"
	try:
		pil_image = Image.open(image_path)
	except Exception as e:
		return str(e)

	return True

def allProjects_combinedPipeline(image_path):
	from PIL import Image

	check = argCheck_allProjects_combinedPipeline(image_path)
	if not check:
		return check
	pil_image = Image.open(image_path)
	rv = {}
	rv["translated_text"] = image_to_text_function(pil_image)[1]
	rv["converted_excel"] = automateMultipleExcelCreation(list(rv["translated_text"].values()),list(rv["translated_text"].keys()))	 
	return rv

if __name__=="__main__":
	from pprint import pprint
	path = r"C:\Users\gprak\Downloads\text-photographed-eng.jpg"
	ans = allProjects_combinedPipeline(path)
	pprint(ans)