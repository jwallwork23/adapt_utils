from moving_fn import *

res = 0.1

filename = 'nx_' + str(res) + '.txt'

moving(res, 9, 0, 1, filename)

moving(res, 9, 0.5, 1, filename)

moving(res, 9, 1, 1, filename)

moving(res, 11, 0, 1, filename)
