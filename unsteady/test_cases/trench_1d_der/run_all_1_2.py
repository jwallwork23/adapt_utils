from moving_fn import *

res = 0.1

filename = 'nx_' + str(res) + '.txt'

moving(res, 3, 0.5, 1, filename)

moving(res, 3, 1, 1, filename)

moving(res, 5, 0, 1, filename)

moving(res, 5, 0.5, 1, filename)

