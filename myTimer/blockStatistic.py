from define import *
class BlockStatistic():
    def __init__(self):
        self.d = {}
        self.lastStart = -1
        self.meansum = 0
        # self.count = 0
    def add(self, startAt, timeDiff):
        self.d[startAt] = timeDiff
    def start(self,):
        self.lastStart = tt()
    def stop(self,):
        timediff = tt() - self.lastStart
        self.d[self.lastStart] = timediff
        self.meansum += timediff
        self.lastStart = -1
    def isRecording(self,):
        return self.lastStart != -1
    def getMean(self,):
        if self.count:
            return self.meansum/self.count
        else:
            return 0
    @property
    def count(self,):
        return self.d.__len__()
    def getData(self,name = ''):
        data = pandas.DataFrame([[self.meansum,self.count,self.getMean()]],[name],['all time use','use count','mean'])
        return data
    def getFullData(self,name = ''):
        l = sorted(self.d.items())
        timeindexs,datalist = zip(*l)
        data = pandas.Series(datalist,timeindexs,None,name)
        return data
    def getAllStatistic(self):
        return self.d
    def __str__(self):return getData()