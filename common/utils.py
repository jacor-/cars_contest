from pylab import *

def multiplot(images, shape):
    figure()
    for i in range(len(images)):
        if i+1 > shape[0]*shape[1]:
            break
        subplot(shape[0], shape[1], 1+i)
        imshow(images[i])