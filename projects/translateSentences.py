from urbans import Translator


def langTranslate(src_sentences,src_grammar,src_to_target_grammar,lang1_to_lang2_dict):
    translator = Translator(src_grammar = src_grammar,
                        src_to_tgt_grammar = src_to_target_grammar,
                        src_to_tgt_dictionary = lang1_to_lang2_dict)
    trans_sentences = translator.translate(src_sentences)
    return trans_sentences
    
def multipleLangTranslate(src_sentences,src_grammarLists,src_to_target_grammarLists,lang1_to_lang2_dictLists):
    for i in range(1,len(src_grammarLists)):
        src_sentences = langTranslate(src_sentences,src_grammarLists[i-1],src_to_target_grammarLists[i-1],lang1_to_lang2_dictLists[i-1])
    return src_sentences


if __name__=="__main__":
    # Source sentence to be translated
    src_sentences = ["I love good dogs", "I hate bad dogs"]

    # Source grammar in nltk parsing style
    src_grammar = """
                    S -> NP VP
                    NP -> PRP
                    VP -> VB NP
                    NP -> JJ NN
                    PRP -> 'I'
                    VB -> 'love' | 'hate'
                    JJ -> 'good' | 'bad'
                    NN -> 'dogs'
                    """

    # Some edit within source grammar to target grammar
    src_to_target_grammar =  {
        "NP -> JJ NN": "NP -> NN JJ" # in Vietnamese NN goes before JJ
    }
    target_to_src_grammar = {
        "NP -> NN JJ": "NP -> JJ NN"  # in Vietnamese NN goes before JJ
    
    }
    # Word-by-word dictionary from source language to target language
    en_to_vi_dict = {
        "I":"tôi",
        "love":"yêu",
        "hate":"ghét",
        "dogs":"những chú_chó",
        "good":"ngoan",
        "bad":"hư"
        }
    vi_to_en_dict= {
        "tôi":"I",
        "yêu":"love",
        "ghét":"hate",
        "những chú_chó":"dogs",
        "ngoan":"good",
        "hư":"bad"
    
    }

    trans_sentences = langTranslate(src_sentences,src_grammar,src_to_target_grammar,en_to_vi_dict)
    print(trans_sentences)
    src_grammarLists = [
        src_grammar,
        src_grammar,
        
    ]
    src_to_target_grammarLists = [
        target_to_src_grammar,
        src_to_target_grammar,
     
        
    ]
    lang1_to_lang2_dictLists = [
        vi_to_en_dict,
        en_to_vi_dict,
       
    ]
    trans_sentences = multipleLangTranslate(src_sentences,src_grammarLists,src_to_target_grammarLists,lang1_to_lang2_dictLists)
    print(trans_sentences)