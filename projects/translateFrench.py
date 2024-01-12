import spacy
from spacy.lang.fr.examples import sentences 

def translate_to_french(sentences,lists = ["text","pos_","dep_"]):
    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(sentences)
    rv = {}
    rv["doc.text"] = doc.text
    for i,token in enumerate(doc):
        rv[f"token-{i}"] = token
        rv[f"token-{i}"] = {}
        for l in lists:    
            try:
                rv[f"token-{i}"][l] = eval(f"token.{l}")
            except:
                rv[f"token-{i}"][l] = f"token.{l}"
    return rv

if __name__=="__main__":
    from pprint import pprint
    fr_sent = translate_to_french("HI I am Prakhar!")
    pprint(fr_sent)