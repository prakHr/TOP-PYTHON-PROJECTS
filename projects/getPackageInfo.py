import subprocess
def get_latest_packages_info():
	subprocess.call(['pip-review'])

def install_latest_packages():
	subprocess.call(['pip-review','--auto'])

def install_interactively_latest_packages():
	subprocess.call(['pip-review','--interactive'])

if __name__=="__main__":
	#get_latest_packages_info()
	install_interactively_latest_packages()