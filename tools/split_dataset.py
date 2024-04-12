import random

root = '/data0/dataset/aapm/ct'
bodies = open(f'{root}/body.txt').readlines()
heads = open(f'{root}/head.txt').readlines()

for i in range(10):
    random.shuffle(bodies)
    random.shuffle(heads)

result = [], [], [], [], []

count = 0
data = bodies + heads
for i in range(len(data)):
    g = i % len(result)
    result[g].append(data[i])

for i in range(5):
    r = result[i]
    for _ in range(10):
        random.shuffle(r)
    f = open(root + f'/data-{i + 1}.txt', 'w')
    for l in r:
        f.write(l)
