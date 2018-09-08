import os
import sys
import random

targetpath = './train/'
destinationpath = './val/'
num_sample = 25

if not os.path.exists(destinationpath):
    os.mkdir(destinationpath)
else:
    print('error already path is exist!!')
    sys.exit(0)

folderlist = os.listdir(targetpath)
print('folder list : {}'.format(folderlist))
for dirpath in folderlist:
    if not os.path.exists(os.path.join(destinationpath, dirpath)):
        os.mkdir(os.path.join(destinationpath, dirpath))
    loadpath = os.path.join(targetpath,dirpath)
    datalist = os.listdir(loadpath)
    samplelist = random.sample(datalist, num_sample)
    for filename in samplelist:
        os.renames(os.path.join(loadpath, filename), os.path.join(destinationpath, dirpath, filename))


print('Sampleing is complete')
