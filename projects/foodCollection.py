from pyfood.utils import Shelf
def getShelf(country_name,month_id,ingredient_list,language,seasonal_fruit_key,seasonal_veg_key):
	shelf = Shelf(region=country_name, month_id=month_id)
	results = shelf.process_ingredients(ingredient_list)
	results2 = shelf.process_ingredients(ingredient_list,lang_dest = language)
	fruits_in_season = shelf.get_seasonal_food(key=seasonal_fruit_key)
	vegetables_in_season = shelf.get_seasonal_food(key=seasonal_veg_key)
	labels = {
		'labels':results['labels'],
		'ingredients_by_taxon':results2['ingredients_by_taxon'],
		'fruits_in_season':fruits_in_season,
		'vegetables_in_season':vegetables_in_season
	}


	return labels
if __name__=="__main__":
	country_name = "France"
	month_id = 0
	ingredient_list = ['apple','kiwi','sugar']
	language = 'FR'
	seasonal_fruit_key = '001'
	seasonal_veg_key = '002'
	rv = getShelf(country_name,month_id,ingredient_list,language,seasonal_fruit_key,seasonal_veg_key)
	print(rv) 