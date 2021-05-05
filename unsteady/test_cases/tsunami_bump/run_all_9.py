from moving_func import *

alpha = 9
nx = 5
ny = 5

filename = 'nx_' + str(nx) + '_alpha_' + str(alpha) + '.txt'

#main(filename, alpha= alpha, mod=0, beta=1, gamma=0, nx=nx, ny=ny)

#main(filename, alpha= alpha, mod=0.5, beta=1, gamma=0, nx=nx, ny=ny)

#main(filename, alpha= alpha, mod=1, beta=1, gamma=0, nx=nx, ny=ny)

#main(filename, alpha= alpha, mod=0, beta=0, gamma=1, nx=nx, ny=ny)

#main(filename, alpha= alpha, mod=0.5, beta=0, gamma=1, nx=nx, ny=ny)

#main(filename, alpha= alpha, mod=1, beta=0, gamma=1, nx=nx, ny=ny)

#main(filename, alpha= alpha, mod=0, beta=1, gamma=1, nx=nx, ny=ny)

main(filename, alpha= alpha, mod=0.5, beta=1, gamma=1, nx=nx, ny=ny)

main(filename, alpha= alpha, mod=1, beta=1, gamma=1, nx=nx, ny=ny)

main(filename, alpha= alpha, mod=0.5, beta=0, gamma=0, nx=nx, ny=ny)

main(filename, alpha= alpha, mod=1, beta=0, gamma=0, nx=nx, ny=ny)
