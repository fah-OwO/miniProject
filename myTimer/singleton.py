class Singleton(object):
#   def __new__(cls):
#     if not hasattr(cls, 'instance'):
#       cls.instance = super(Singleton, cls).__new__(cls)
#     return cls.instance
    # instance = {}
    # def __new__(cls):
    #     if cls not in cls.instance:
    #         cls.instance[cls] = super(Singleton, cls).__new__(cls)
    #     return cls.instance[cls]
    instance = None
    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

class BorgSingleton(object):
    _shared_borg_state = {}
    
    def __new__(cls, *args, **kwargs):
        obj = super(BorgSingleton, cls).__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_borg_state
        return obj

def testSingleton():
    class dummy(Singleton):
        def __init__(self):
            self._t = tt()
            self.d = {}
            self.s = self.start
            self.s_ = self.stop
            self.p = print
            self.lastL = None
        def setS(self,s):self.s = s
        pass
    a= Singleton()
    b= Singleton()
    c= dummy()
    d= dummy()
    l = [a,b,c,d]
    for i in l:
        print(*[int(j == i) for j in l])
    c.setS(10)
    print(c.s)
    print(d.s)
    d.setS(20)
    print(c.s)
    print(d.s)

if __name__ == '__main__':
    testSingleton()
    pass