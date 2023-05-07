from define import *
_t = tt()
def t(prin = True):
    global _t
    __t = tt()
    out = __t - _t
    if prin:
        if type(prin) is str:
            print(prin,out)
        else:
            print(out)
    _t = tt()
    return out

    