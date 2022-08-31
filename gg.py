import glob
files = glob.glob('*.csv')
print(files)
num = 1
for file in files:
    if('submit' in file):
        if(str(num) in file):
            num += 1
    
print(num)