from eztimer import *
from mainTimer import MainTimer
from decorator import time_it
def testDec():
    @time_it
    def a():
        for i in range(1e6):
            i+=1
    MainTimer().show()

if __name__ == '__main__':
    testDec()
    pass
        