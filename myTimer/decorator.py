from mainTimer import MainTimer
def time_it(func):
    t = MainTimer()
    def inner1(*args, **kwargs):
        t.start(func)
        returned_value = func(*args, **kwargs)
        t.stop(func)
        return returned_value
         
    return inner1

    
def testDec():
    @time_it
    def a():
        for i in range(int(1e6)):
            i+=1
    a()
    MainTimer().show()

if __name__ == '__main__':
    testDec()
    pass