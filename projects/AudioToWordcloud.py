def audio_to_wordcloud(mp3_input,model_type = "base",show_image = True):
	import whisper
	from PIL import Image
	def text_to_freqDict(text):
		SPACE = " "
		text_list = text.split(SPACE)
		rv = {}
		for text in text_list:
			rv[text] = rv.get(text,0)+1
		return rv

	def freqDict_to_wordCloud(freq_dict):
		from wordcloud import WordCloud
		import base64
		from io import BytesIO
		wc = WordCloud().generate_from_frequencies(frequencies=freq_dict)
		wc_img = wc.to_image()
		return wc_img

	model = whisper.load_model(model_type)
	result = model.transcribe(mp3_input)
	text_output = result["text"]
	freq_dict = text_to_freqDict(text_output)
	wc_img = freqDict_to_wordCloud(freq_dict)
	if show_image==True:
		wc_img.show()
	return wc_img

if __name__=="__main__":
	mp3_input = r"C:\Users\gprak\Downloads\DiscUdemy Courses\download1.wav"
	wc_img = audio_to_wordcloud(mp3_input)
	print(wc_img)