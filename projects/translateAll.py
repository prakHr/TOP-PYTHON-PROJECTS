import spacy
from spacy.lang.fr.examples import sentences 

def get_postags(
        sentences,
        language="fr",
        lists = ["text","pos_","dep_"]
    ):
    nlp = spacy.load(f"{language}_core_news_sm")
    doc = nlp(sentences)
    rv = {}
    rv["doc.text"] = doc.text
    for i,token in enumerate(doc):
        rv[f"token-{i}"] = {}
        for l in lists:    
            try:
                rv[f"token-{i}"][l] = eval(f"token.{l}")
            except:
                rv[f"token-{i}"][l] = f"token.{l}"
    return rv

if __name__=="__main__":
    from pprint import pprint
    sent = "Hi I am Prakhar and My partner is Jin Tian!"
    default_fr_sent = get_postags(sent)
    pprint(default_fr_sent)

    es_sent = get_postags(sent,language="es")
    pprint(es_sent)

    