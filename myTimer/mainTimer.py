from define import *
from blockStatistic import BlockStatistic
# from singleton import Singleton
def _statisticgetFullDatafunc(statistic,name):return statistic.getFullData(name)
        
class TimerDB():
    instance = None
    def __new__(self, *args, **kwargs):
        if self.instance is None:
            self.d = {}
            self.instance = super().__new__(self, *args, **kwargs)
        return self.instance

class MainTimer(TimerDB):
    def __init__(self):
        self._t = tt()
        self.s = self.start
        self.s_ = self.stop
        self.p = print
        self.lastL = None
    def __call__(self,*args,**kwargs):return t(*args,**kwargs)
    
    #   t.start(key) start recording timer (auto created)
    #   t.stop(key)  stop  recording timer (if no key: print) won't throw error
    def start(self,key):
        if key not in self.d:
            self.d[key] = BlockStatistic()
        self.d[key].start()
    def stop(self,key):
        if key not in self.d:print(' no key',key,'in',d)
        self.d[key].stop()
        
    #   t.dataframe()
    #   summary
    @property
    def dataframe(self,):
        df = pandas.DataFrame()
        for name,statistic in self.d.items():
            df = pandas.concat([df,statistic.getData(name)])
        return df
    #   t.fullData()
    #   return full data when it start?, how long it last?
    @property
    def fullData(self,):
        return [statistic.getFullData(name) for name,statistic in self.d.items()]

        import multiprocessing as mp
        # mp.freeze_support()
        pool = mp.Pool(mp.cpu_count())
        # func = lambda statistic,name:statistic.getFullData(name)
        data = pool.starmap(_statisticgetFullDatafunc,self.d.items())
        return data

    def showAll(self,):
        df = self.dataframe
        self.p(df)
        return df
    def show(self,key = None):
        if key is None:return self.showAll()
        df = self.d[key].getData(key)
        self.p(df)
        return df
    def __str__(self):
        return str(self.dataframe)
    # use t.a = key
    #   use to start and stop the key
    #   if it already start then stop,
    #   if already stop then start
    @property
    def a(self):return self.__str__()
    @a.setter
    def a(self,key):
        if key not in self.d:
            self.d[key] = BlockStatistic()
        block = self.d[key]
        if block.isRecording():
            block.stop()
        else:
            block.start()
    # use t.l = key
    #   stop the last key that was called using t.l
    #   then start timer with key
    @property
    def l(self):
        if self.lastL!=None:
            self.stop(self.lastL)
        self.lastL = None
    @l.setter
    def l(self,key):
        if self.lastL!=None:
            self.stop(self.lastL)
        self.lastL = key
        self.start(key)

    #   save and read file
    #   input: filename(string)
    def save(self, filename):
        import pickle
        with open(filename,'wb') as f:
            pickle.dump(self, f)
    #   dont use: error
    def read(self, filename):
        import pickle
        f = open(filename,'rb')
        with open(filename, 'rb') as f:
            self = pickle.load(f)

def Ttest():
    t = MainTimer()
    for i in range(10):
        t.s('loop')
        for i in range(1000):
            t.s('x')
            t.s_('x')
        t.s_('loop')
    t.show()
    print(t)
    t.show()
    file_name = r'./TimerTest.txt'
    t.save(file_name)
    t.s('after dump')
    print('==========')
    import pickle
    pickle.load(open(file_name, 'rb')).show()
    print('==========')
    t_read = MainTimer()
    t_read.read(file_name)
    (t_read).show()
    print('==========')
    t.s_('after dump')
    (t).show()
if __name__ == '__main__':
    Ttest()
        