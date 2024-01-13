from ftfy import fix_encoding,fix_text
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

if __name__=="__main__":
	garbage_text = 'âœ” No problems'
	ans = garbageCleaner(garbage_text)
	print(ans)

	garbage_texts = ['âœ” No problems','IL Y MARQUÉ…','P&EACUTE;REZ']
	ans = garbagesCleaner(garbage_texts)
	print(ans)
