from chinese import ChineseAnalyzer
import random
def translate_chinese(chinese_text):
    analyzer = ChineseAnalyzer()
    result = analyzer.parse(chinese_text,traditional = True)
    rv = ""
    space = " "
    for token in result.tokens():
        shenme = result[token]
        shenme_info = shenme[0]
        rv+= random.choice(shenme_info.definitions)
        rv+=space
    return rv

if __name__=="__main__":
    chinese_text = '我喜歡這個味道'
    rv = translate_chinese(chinese_text)
    print(rv)