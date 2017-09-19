"""
SVM_GUI
GUI for presenting linear SVM with matplotlib and PyQT

Stefan Wong 2017
"""

#from PyQt4.QtCore import *
#from PyQt4.QtGui import *

#import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
#from matplotlib.figure import Figure
import numpy as np

# Debug
#from pudb import set_trace; set_trace()


#class SVMGUI(QMainWindow):         # TODO ; worry about QT later...
class SVMGUI(object):
    def __init__(self, parent=None):
        #self.setWindowTitle('Linear SVM interactive demo')


        # Test the canvas
        self.dpi = 100
        self.ax = []
        self.fig = plt.figure((6.0, 4.0), dpi=self.dpi)         # TODO ; main_fig?
        #self.canvas = FigureCanvas(self.fig)       # TODO : this is some QT thing....

    #def plot(self, ..., ax=None, **kwargs):

    #    if ax is None:
    #        ax = gca()

    #    ax.plot(..., **kwargs)

    def draw_test(self):

        # Sample data
        dt = 0.001
        t = np.arange(0.0, 10.0, dt)
        r = np.exp(-t[:1000]/0.05)
        x = np.random.randn(len(t))
        s = np.convolve(x, r)[:len(x)] * dt

        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set_title("TEST TITLE")
        self.ax.plot([1, 2, 3, 4])
        self.ax.set_ylabel('some numbers')





if __name__ == "__main__":

    gui = SVMGUI()
    gui.draw_test()
