from moving_fn import *

res = 0.1

filename = 'nx_' + str(res) + '.txt'

moving(res, 13, 1, 1, filename)

moving(res, 15, 1, 1, filename)

moving(res, 15, 0.5, 1, filename)

moving(res, 15, 0, 1, filename)
