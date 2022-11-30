import csv
from itertools import chain
import time


path = f'./SampleData/'
with open(path+ 'Father/' + 'tags.csv', newline='') as f:
    reader = csv.reader(f)
    tags = list(reader)
tags = list(map(float, list(chain.from_iterable(tags))))

for ut in tags:
    dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ut) )
    print(dt)