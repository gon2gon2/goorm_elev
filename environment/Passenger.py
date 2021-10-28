import random

class Passenger(object):
    '''
    The main concern of this problem is passenger transportation.
    The number of passenger is randomly generated on all floor.
    And destination of all passenger us also randomly generated.
    '''
    def __init__(self, now_floor : int, max_floor : int, dest : int):
        self.now_floor = now_floor
        self.max_floor = max_floor
        self.up = int(dest > now_floor)
        self.dest = dest
    
                                      
    ### state로 넘겨줄 때 쓰는 거
    def get_dest(self) -> int:
        return self.dest
    
    def get_dir(self) -> int:
        return self.up