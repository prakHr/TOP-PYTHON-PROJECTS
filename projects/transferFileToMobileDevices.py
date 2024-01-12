import boat

def tranfer_to_mobile_devices(filename , url = False, qr = False):
	import subprocess
	if url == True:
		subprocess.call(['boat',filename])
	if qr  == True:
		subprocess.call(['boat',filename,'--qr'])

if __name__=="__main__":
	filename = r"C:\Users\gprak\Downloads\planning_utils.py"
	tranfer_to_mobile_devices(filename,url = True)