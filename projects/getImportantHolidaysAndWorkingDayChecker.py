from datetime import date
import datetime
import workalendar
def get_country_calender(country_name='europe',state_name = 'France',year = 2012):
	from importlib import import_module
	moduleName = f"from workalendar.{country_name} import {state_name}"
	exec(moduleName)
	cal = eval(f"{state_name}()")
	holiday = cal.holidays(year)
	print(f"is it working day today? {cal.is_working_day(datetime.datetime.now().date())}")
	return holiday
	
if __name__=="__main__":
	h = get_country_calender()
	print(h)