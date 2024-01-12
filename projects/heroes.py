#import module
from bmwwAPI import bmww
import copy

#create api object
api = bmww()



def get_ten_or_less_comicstories_serieswise(mini,superhero,step,all_hero_keys):
  """
  Get <=10 comic stories according to starting series number
  Params:-
    mini : starting series number,
    superhero : superhero name,
    step : no of steps <=10,
    all_hero_keys : list of all hero key
  Returns:-
    stories : dict of all fav info about comic heros
  """
  print(f"series starts from no. {mini} to {mini+min(step,10)} with superhero {superhero}")
  stories = {}
  for id in range(mini,mini+min(step,10)):
    try:
      print(id)
      work = api.work(id)  
      work_info = []
      work_infoo = {}
      for all_hero_key in all_hero_keys:
        k = f"{all_hero_key}"
        v = f"work.{all_hero_key}"
        try:
          work_infoo[k] = eval(v)
        except:
          work_infoo[k] = k
      if superhero in (work.title):
        stories[work.title] = work_infoo 
      
    except Exception as e:
      print(e)
      continue
  return stories

if __name__=="__main__":
  step = int(input("Enter steps:-"))
  mini = int(input("Enter series starting number:-"))
  superhero = input("Enter superhero name:-")
  all_hero_keys = [
      "title",         
      "author",
      "reviews"
  ]
  stories = get_ten_or_less_comicstories_serieswise(mini,superhero,step,all_hero_keys)
  print(stories)