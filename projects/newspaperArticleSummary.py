import newspaper
from newspaper import Article

def download_article(url):
	article = Article(url)
	article.download()
	article.parse()
	
def build_newspaper(url):
	a = Article(url)
	a.download()
	a.parse()
	return {
		"texts":a.text,
		"titles":a.title
	}
if __name__=="__main__":
	url ='http://fox13now.com/2013/12/30/new-year-new-laws-obamacare-pot-guns-and-drones/'
	rv = build_newspaper(url)
	print(rv)