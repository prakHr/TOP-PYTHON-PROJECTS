import fire

def hello(name):
  return "Hello %s!" % name

def addition(a,b):
  return a+b
  
def check_correct_filepath(filepath):
  import os
  return os.path.exists(filepath)

def is_int_present(string):
  import re
  check = re.findall(r'\d+', str(string))
  return len(check)!=0

def is_substr_present(substr,string):
  import re
  check =  re.search(substr,string)
  return len(check.group())!=0

def is_pattern_present(pattern,string,substitute="Java"):
  import re
  print("span match:-")
  match = re.search(str(pattern),str(string))
  print(match.span())

  print("all matches:-")
  matches = re.findall(pattern,string)
  print(matches)

  print("splits:-")
  splits  = re.split(pattern,string)
  print(splits)

  print("substituted_string:-")
  substituted_string = re.sub(pattern,substitute,string)
  print(substituted_string)

  print("positive lookahead match:-")
  pos_pattern = fr"(?<={pattern}\s)\w+"
  poslookahead_match = re.search(pos_pattern,string)
  print(poslookahead_match.group())

  print("negative lookahead match:-")
  neg_pattern = fr"(?<!{pattern}\s)\w+"
  neglookahead_match = re.search(neg_pattern,string)
  print(neglookahead_match.group())

  return len(match.group())!=0

if __name__ == '__main__':
  namee = input("Enter function param name:-")
  if namee=="hello":
    name = input("enter name!")
    fire.Fire(hello(name))
  if namee=="addition":
    a = int(input("enter a!"))
    b = int(input("enter b!"))
    fire.Fire(addition(a,b))
  if namee=="chath":
    filepath = input("enter filepath!")
    fire.Fire(check_correct_filepath(filepath))
  if namee=="isint":
    string = input("enter string!")
    fire.Fire(is_int_present(string))
  if namee=="substrpresent":
    substr = input("enter substr!")
    string = input("enter string!")
    fire.Fire(is_substr_present(substr,string))
  if namee=="patternpresent":
    pattern = input("enter pattern!")
    string = input("enter string!")
    fire.Fire(is_pattern_present(pattern,string))