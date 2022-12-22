
def init():
    global RESCALE, N_, M_, x0_, y0_, dx_, dy_, fps_
    RESCALE = 1

    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker (in original size)
    dx_, dy_: the horizontal and vertical interval between adjacent markers (in original size)
    fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
    """

    ## R1.5. small dots (imgw,imgh) (240,320)
    # N_ = 14
    # M_ = 10
    # fps_ = 30
    # x0_ = 11 / RESCALE
    # y0_ = 9 / RESCALE
    # dx_ = 33 / RESCALE
    # dy_ = 26 / RESCALE

    ## R1.5. small dots (imgw,imgh) (480,640)
    # N_ = 14
    # M_ = 10
    # fps_ = 30
    # x0_ = 52 / RESCALE
    # y0_ = 16 / RESCALE
    # dx_ = 43 / RESCALE
    # dy_ = 43 / RESCALE

    ## gel number #58. on R1 with small dots. image size (h,w) (240,320)
    # N_ = 10
    # M_ = 14
    # fps_ = 30
    # x0_ = 13 / RESCALE
    # y0_ = 20 / RESCALE
    # dx_ = 21 / RESCALE
    # dy_ = 21 / RESCALE

    ##  on GS mini with small dots. image size (h,w) (240,320)
    N_ = 7
    M_ = 9
    fps_ = 25
    x0_ = 55 / RESCALE
    y0_ = 22 / RESCALE
    dx_ = 31 / RESCALE
    dy_ = 32 / RESCALE