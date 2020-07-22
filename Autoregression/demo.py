import subprocess 
import os 

s = subprocess.check_output("python3 predict_cases.py", shell = True) 
print(s.decode("utf-8"))

s = subprocess.check_output("python3 predict_deaths.py", shell = True) 
print(s.decode("utf-8"))
