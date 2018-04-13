import time
class Timer:
    def __init__(self):
        self.time1 = 0
        self.time2 = 0
        self.diff  = 0
        
    def start(self):
        self.time1 = time.time()
    
    def stop(self):
        self.time2 = time.time()
        self.diff = self.time2 - self.time1
        
    def report(self, niter = 1):
        val = self.diff / niter
        return val // 3600, val // 60 % 60, val % 60
