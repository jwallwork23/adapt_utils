from moving_fn import *

res = 0.1

filename = 'nx_' + str(res) + '.txt'


moving(res, 5, 1, 1, filename)

moving(res, 7, 0, 1, filename)

moving(res, 7, 0.5, 1, filename)

moving(res, 7, 1, 1, filename)
