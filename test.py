from time import sleep
import sys

n= 10

for i in range(n):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[{:{}}] {:.1f}%".format("="*i, n-1, (100/(n-1)*i)))    
    sys.stdout.flush()
    sleep(.5)


