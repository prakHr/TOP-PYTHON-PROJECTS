def reconstruct_text(text,dst_language = "en"):
	from deep_translator import GoogleTranslator
	from langdetect import detect
	from fuzzywuzzy import fuzz,process
	src_language = detect(text)
	constructed_text   = GoogleTranslator(source=src_language, target=dst_language).translate(text)
	reconstructed_text = GoogleTranslator(source=dst_language, target=src_language).translate(constructed_text)
	rv = {}
	rv["text"] = text 
	rv["constructed_text"] = constructed_text
	rv["reconstructed_text"] = reconstructed_text
	rv["best_metric"] = float((fuzz.token_set_ratio(text,reconstructed_text))/100)
	return rv

if __name__=="__main__":
	text = 'नमस्ते मैं प्रखर हूँ!'
	ans = reconstruct_text(text,"si")
	print(ans)