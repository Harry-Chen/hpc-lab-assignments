import sys

prod = 1.
total = 0.
cnt = 0.

for l in sys.stdin:
	if not l.startswith('Size'):
		continue
	fl = float(l.split(': ')[2].split(' ')[0])
	total += fl
	prod *= fl
	cnt += 1
print('Geo %.2f Arith %.2f' % (prod ** (1. / cnt), total / cnt))

