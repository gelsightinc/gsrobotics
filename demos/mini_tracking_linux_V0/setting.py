
def init():
    global RESCALE, N_, M_, fps_
    RESCALE = 1

    """
    N_, M_: the row and column of the marker array
    fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
    """
    ##  on GS mini with small dots. image size (h,w) (240,320)
    N_ = 7
    M_ = 9
    fps_ = 25
