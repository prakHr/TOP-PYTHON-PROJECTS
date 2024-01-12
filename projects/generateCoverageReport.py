import subprocess
def generate_report(python_file):
  subprocess.call(["python","-m","coverage","html",python_file])

if __name__=="__main__":
  python_file = "translateSentences.py"
  generate_report(python_file)