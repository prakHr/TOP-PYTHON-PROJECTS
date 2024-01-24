def image_feature_extraction(image_path):
	"""
	{'Colorfulness_HSV': 0.01738628417456691,
	 'Colorfulness_RGB': 0.012947419195837433,
	 'Colors': [['Aqua', 0.0],
	            ['Black', 2.0774999999999997],
	            ['Blue', 0.0],
	            ['Fuchsia', 0.0],
	            ['Gray', 6.987500000000001],
	            ['Green', 0.0],
	            ['Lime', 0.0],
	            ['Maroon', 0.0],
	            ['Navy', 0.0],
	            ['Olive', 0.0],
	            ['Purple', 0.0],
	            ['Red', 0.0],
	            ['Silver', 4.804166666666667],
	            ['Teal', 0.0],
	            ['White', 86.13083333333333],
	            ['Yellow', 0.0]],
	 'Faces': (),
	 'Number_of_Faces': 0,
	 'Number_of_Images': 0,
	 'Symmetry_QTD': 77.68595041322314,
	 'Text': 191,
	 'TextImageRatio': 1.0,
	 'VC_quadTree': 607,
	 'VC_weight': 200739,
	 'brightness_BT601': 0.8870466890186475,
	 'brightness_BT709': 0.8870466905411412,
	 'imageArea': 0,
	 'textArea': 196287}
	"""
	import pyaesthetics
	results = pyaesthetics.analysis.analyzeImage(image_path, method="complete")
	return results

def text_feature_extraction(text_path):
	"""
	[{"txt":"Sears\\nwhy you want to be part of the program and why you want to '
	 'join the company.(Done)\\n\\nTell me about an experience that showed your '
	 'integrity.(Done)\\n\\nB2C websites\\nThey asked me to verbally walkthrough '
	 'the selection through checkout on a site like Amazon when purchasing an '
	 'item.\\n\\nWhat logical process would you pursue to consider whether it '
	 'would be profitable to put one of our product \\nlines in another retail '
	 'store?\\n\\nWhat do you know about Agile?(Done)\\n\\nWHich phase of SDLC is '
	 'important for BA?\\n\\nHow would you handle an unjustified Change '
	 'request?(Done)\\n\\nRoleplay - ask requirement questions to a '
	 'stakeholder\\n\\nWhat are some challenges you foresee in this '
	 "role?\\n\\nTell me about a time when you didn't agree with the rest of the "
	 'group and how you went about it.(Done)\\n\\nTell me about a time when you '
	 'had to deliver bad news.(Done)\\n\\nWhere do you see yourself in 5 '
	 'years.(Done)\\n\\nTell me about a time when you assumed a leadership '
	 'position and made a positive impact.(Done)\\n\\nDescribe a situation where '
	 'you had a disagreement with someone and what happened.(Done)\\n\\nTell me a '
	 'little bit about yourself.(Done)\\n\\nIf you had one item at the '
	 'distribution center and had to decide between two stores where to send it '
	 'to, how would you decide?\\n\\nGive me an example of a time where you had to '
	 'make a difficult decision\\n\\nTell me about a time where you faced an '
	 'ethical dilemma in the workplace and how you handled it. Also the group '
	 'activity was difficult.\\n\\nTell me about a time when you faced adversity. '
	 'How did you handle it and what was the outcome?\\n\\nTechnical questions '
	 'regarding python data structure\\n\\nIntuit\\n1. describe your current '
	 "role\\n2. why do you think you are a good fit for the role\\n3. what's your "
	 'plan for the next 5 '
	 'years\\n\\n","word_count":281,"char_count":1725,"avg_word_length":4.5653594771,"stopwords_count":150,"hashtags_count":0,"links_count":0,"numerics_count":2,"user_mentions_count":0}]'
 	"""
	import textfeatures
	import pandas as pd
	from textfeatures import word_count,char_count,avg_word_length,stopwords_count,stopwords,hashtags_count,hashtags,links_count,links,numerics_count,user_mentions_count,user_mentions,clean
	f = open(text_path, "r")
	txt = f.read()
	results = pd.DataFrame()
	results['txt'] = [txt]
	fn_names = ["word_count",
		"char_count",
		"avg_word_length",
		"stopwords_count",
		"hashtags_count",
		"links_count",
		"numerics_count",
		"user_mentions_count"
	]
	for fn_name in fn_names:
		x = fn_name
		ans = eval(f'{fn_name}(results,"txt","{x}")')
		
	return results.to_json(orient= 'records')

if __name__=="__main__":
	image_path = "/mnt/c/Users/gprak/downloads/projects/data/handwritten_image.png"
	rv = image_feature_extraction(image_path)
	from pprint import pprint
	pprint(rv) 

	text_path = "/mnt/c/Users/gprak/downloads/projects/data/ExpectedBehavioural.txt" 
	rv = text_feature_extraction(text_path)
	from pprint import pprint
	pprint(rv) 
