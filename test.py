import pylab
import numpy

x = numpy.linspace(0, 200)  # 100 linearly spaced numbers
y = numpy.sin(x)/x  # computing the values of sin(x)/x
s = 1/(1 + (128/x)**100)
# compose plot
pylab.plot(x, s)  # sin(x)/x
# pylab.plot(x, y, 'co')  # same function with cyan dots
# pylab.plot(x, 2*y, x, 3*y)  # 2*sin(x)/x and 3*sin(x)/x
pylab.show()  # show the plot
