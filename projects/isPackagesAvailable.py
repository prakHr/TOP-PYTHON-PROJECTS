import subprocess 

def isPackagesAvailable(list_of_package_names,show_on_terminal=False,file_name="results"):
    if show_on_terminal:
        subprocess.call("pynamer "+" ".join(list_of_package_names))
        return
    subprocess.call("pynamer "+" ".join(list_of_package_names)+" -o "+file_name)

if __name__=="__main__":
    isPackagesAvailable(["boat","steve","pyAlgebra"],True)