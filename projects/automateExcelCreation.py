from workbook import Workbook
from faker import Faker


def automateExcelCreation(wb,data,sheet_name):
	wb.write_sheet(data,sheet_name,print_to_screen=False)
	return wb 


def automateMultipleExcelCreation(list_of_data,list_of_sheet_name,output_csv_name):
	wb = Workbook()
	for data,sheet_name in zip(list_of_data,list_of_sheet_name):
		wb = automateExcelCreation(wb,data,sheet_name)
	wb.save(output_csv_name)

if __name__=="__main__":
	fake = Faker()
	names = [fake.name() for _ in range(100)]
	addresses = [fake.address() for _ in range(100)]
	texts = [fake.text() for _ in range(100)]

	data1 = [['a','b'],[1,2],[3,4]]
	data2 = [names,addresses,texts]
	
	list_of_data = [data1,data2]
	list_of_sheet_name = ['sheet1','sheet2']
	output_csv_name = 'output.xls'
	
	automateMultipleExcelCreation(list_of_data,list_of_sheet_name,output_csv_name)