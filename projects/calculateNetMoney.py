from decimal import Decimal
import itertools 
import operator

class Money:
	def __init__(self,money,currency_type,id):
		self.money = money
		self.currency_type = currency_type
		self.id = id

	def quantize(self):
		self.money = float(self.money)
		return self.money


def calculate_net_total_money(give_price_list,take_price_list):
	give_price_list_C = give_price_list
	take_price_list_C = take_price_list

	give_price_list = [g.quantize() for g in give_price_list]
	take_price_list = [t.quantize() for t in take_price_list]
	
	result1 = list(itertools.accumulate(give_price_list,operator.add))[-1]
	result2 = list(itertools.accumulate(take_price_list,operator.add))[-1]

	st1 = set([money.id for money in give_price_list_C])
	st2 = set([money.id for money in take_price_list_C])

	lenst = len(st1.union(st2))
	
	x = (result1-result2)/(lenst)
	x = float(x)
	
	return x

if __name__=="__main__":
	give_price_list = [Money(20.123, 'EUR',1),Money(20, 'EUR',2),Money(20, 'EUR',3)]
	take_price_list = [Money(10, 'EUR',11),Money(1000, 'EUR',22),Money(20, 'EUR',33)]
	ans = calculate_net_total_money(give_price_list,take_price_list)
	print(ans)