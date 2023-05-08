import numpy as np
import random

def grid_spacing(pts):
    npts = pts.shape[0]
    # Determine approximate grid spacing
    dists = np.zeros(2*npts)
    for i in range(npts):
        p = pts[i,:]
        dst = np.sqrt( (pts[:,1]-p[1])**2 + (pts[:,0]-p[0])**2 )
        srtd = np.sort(dst)
        dists[2*i] = srtd[1]
        dists[2*i+1] = srtd[2]
    gsp = np.median(dists)
    return gsp


def find_neighbors(pts, p, gsp):
    b = False
    p1 = np.array([])
    p2 = np.array([])
    dlow = gsp * 0.8
    dhigh = gsp * 1.2

    dst = np.sqrt((pts[:,1] - p[1])**2 + (pts[:,0] - p[0])**2)
    inds = np.where((dst > dlow) & (dst < dhigh))[0]
    if len(inds) < 3:
        return b, p1, p2
    p1 = pts[inds[0],:]

    u = p1 - p
    u = u / np.linalg.norm(u)
    # assuming `pts`, `inds`, and `p` are numpy arrays
    v = pts[inds[1:],:] - np.tile(p[np.newaxis,:], (len(inds) - 1,1))

    row_norms = np.linalg.norm(v, axis=1)
    v = v / row_norms[:, np.newaxis]

#    norms = np.sqrt(np.sum(v ** 2, axis=1))
#    v = v.T / (np.tile(norms, (2,1)))

    dt = np.minimum(np.maximum(np.dot(u, v.T), -1), 1)

    ix = np.argmin(dt)
    a = dt[ix]
    p2 = pts[inds[ix+1],:]
    if a < 0.1:
        b = True
        p0 = p

    v = p2 - p
    v = v / np.linalg.norm(v)

    # Which one is closest to horizontal?
    if np.abs(v[0]) < np.abs(u[0]):
        temp = p1.copy()
        p1 = p2.copy()
        p2 = temp

    return b, p1, p2


def find_p0(pts, gsp):
    np = pts.shape[0]

    found = False
    p0 = pts[0,:]
    for i in range(np):
        # Pick a random point
        rx = random.randint(0, np-1)
        p = pts[rx,:]
        v, p1, p2 = find_neighbors(pts, p, gsp)
        if v:
            p0 = p
            found = True
            break
    if not found:
        raise ValueError('did not find central point')

    return p0


def fit_grid(pts, gsp=None, p0=None):

    settings = {"maxitr": 100}

    npts = pts.shape[0]

    if gsp is None:
        gsp = grid_spacing(pts)

    if p0 is None:
        p0 = find_p0(pts, gsp)

    v, p1, p2 = find_neighbors(pts, p0, gsp)

    x = p1[0] - p0[0]
    y = p1[1] - p0[1]
    th = np.arctan2(y, x)

    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    Mpts = (pts - np.tile(p0, (npts, 1)))
    gpts = Mpts.dot(R)

    gpts[:,0] = np.round(gpts[:,0]/gsp)*gsp
    gpts[:,1] = np.round(gpts[:,1]/gsp)*gsp

    x1 = gpts[:,0].reshape(-1,1)
    x2 = gpts[:,1].reshape(-1,1)
    y1 = pts[:,0].reshape(-1,1)
    y2 = pts[:,1].reshape(-1,1)

    s = 1.0
    u = np.array([-th, p0[0], p0[1], s]).reshape(-1,1)

    alpha = np.inf
    itr = 100
    while alpha > 1e-7 and itr < settings["maxitr"]:

        th = u[0]
        b  = u[1:3]

        Jx = np.zeros((npts,4))
        Jy = np.zeros((npts,4))

        jix = np.cos(th)*x1 - np.sin(th)*x2
        jiy = np.sin(th)*x1 + np.cos(th)*x2

        fx = s*jix + b[0] - y1
        fy = s*jiy + b[1] - y2

        Jx[:,0] = s* jix[:,0]
        Jx[:,1] = 1
        Jx[:,3] = jix[:,0]

        Jy[:,0] = s*jix[:,0]
        Jy[:,2] = 1
        Jy[:,3] = jiy[:,0]

        J = np.vstack((Jx, Jy))
        d = np.vstack((fx, fy))

        h = np.linalg.lstsq(J, d, rcond=None)[0]

        u = u + h

        alpha = np.sqrt(np.sum(h**2))

        itr = itr + 1

    th = u[0][0]
    b = u[1:3]

    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ds = np.diag([s,s])
    A = ds.dot(R)

    wpts = gpts.dot(A) + (np.tile(b, npts)).T
    dst = np.sqrt( (wpts[:,0]-pts[:,1])**2 + (wpts[:,1]-pts[:,0])**2 )

    #if wpts is not None and gpts is not None:
    #    import matplotlib.pyplot as plt

    #    fig, ax = plt.subplots()
    #    ax.plot(pts[:,1], pts[:,0], 'b.')
    #    ax.plot(wpts[:,1], wpts[:,0], 'r.')
    #    #q = ax.quiver(pts[0, :], pts[1, :], wpts[0, :] - pts[0, :], wpts[1, :] - pts[1, :], scale=0.5, color='k')
    #    plt.title(f"average error {np.mean(dst):.3f}")
    #    ax.plot(p0[1], p0[0], 'ko', markersize=10, linewidth=2)
    #    ax.set_aspect('equal')
    #    plt.show()

    # FIX THIS, rotating by 180 so wpts match gridw
    # One of the rotations is wrong, fix here for now
    R = np.array([[np.cos(np.pi), np.sin(np.pi)], [-np.sin(np.pi), np.cos(np.pi)]])
    gridw = gpts.dot(R)

    #print(f"average error: {np.mean(dst):.3f}")

    return wpts, gridw
