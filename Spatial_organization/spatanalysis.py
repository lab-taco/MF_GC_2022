import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Pairwise distance between two points
def pair_dist(pt1, pt2) :
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

# Distance between a point and a vector of points
def dist_vec (v_pts, pt) :
    return np.sqrt((v_pts[:,0]-pt[0])**2 + (v_pts[:,1]-pt[1])**2)

# Distance matrix
def dist_mat (crd):
    num_of_ptcl = np.shape(crd)[0]
    d_mat = np.zeros( (num_of_ptcl, num_of_ptcl) )

    for i in range(num_of_ptcl-1) :
        for j in range(num_of_ptcl) :
            
            d_mat[i,j] = d_mat[j,i] = pair_dist(crd[i,:], crd[j,:])

    return d_mat

# Distance matrix for 2 different sets of points
def dist_mat_hetero (crd1, crd2):
    
    ptnum1 = np.shape(crd1)[0]
    ptnum2 = np.shape(crd2)[0]
    d_mat = np.zeros( (ptnum1, ptnum2) )
    
    for i in range(ptnum1) :
        for j in range(ptnum2) :
            d_mat[i,j] = pair_dist(crd1[i,:], crd2[j,:])
            
    return d_mat

# Generate a convex hull from a set of points
def hull_gen(crd) :
    h=ConvexHull(crd)
    return [(crd[x,0], crd[x,1]) for x in h.vertices]

# Generate a random point in a convex hull
def poisproc2d(arena, num_of_pt) :
    pol_arena = Polygon(arena)
    x_arena = np.array(arena)[:,0]
    y_arena = np.array(arena)[:,1]

    minx = np.min(x_arena)
    miny = np.min(y_arena)
    maxx = np.max(x_arena)
    maxy = np.max(y_arena)
    i=1

    generated_x = np.zeros((num_of_pt,1))
    generated_y = np.zeros((num_of_pt,1))
    
    while i<=num_of_pt :
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if pol_arena.contains(Point(x,y)) :
            generated_x[i-1] = x
            generated_y[i-1] = y
            i+=1
    
    return np.hstack((generated_x,generated_y))

# Estimation of G function
def G_fn(crd) :
    d_mat = dist_mat(crd)
    nn_list = np.zeros(np.shape(crd)[0])
    
    for i in range(np.shape(crd)[0]) :
        dist_tmp = d_mat[i,d_mat[i,:].nonzero()]
        nn_list[i] = np.min(dist_tmp)
    
    cdf = ECDF(nn_list)
    
    return nn_list, cdf

# Estimation of F function
def F_fn(h, crd, pt_num) :
    
    coord_points = poisproc2d(h, pt_num)
    d_mat = dist_mat_hetero(coord_points, crd)
    nn_list = np.amin(d_mat, axis=1)
    cdf = ECDF(nn_list)
    
    return nn_list, cdf

# Estimation of K function
def K_fn(h, crd) :

    d_mat = dist_mat(crd)
    exprnum = np.shape(crd)[0]
    arena_poly = Polygon(h)
    arena_area = arena_poly.area
    lmbda = exprnum/arena_area
    
    winv_mat = np.zeros((exprnum, exprnum))

    for i in range(exprnum) :
        for j in range(exprnum) :
            if i != j :
                dij = d_mat[i,j]
                circle = Point(crd[i,0],crd[i,1]).buffer(dij)
                
                if arena_poly.contains(circle) :
                    winv_mat[i,j] = 1
                else :
                    h_rest = arena_poly.difference(circle)
                    h_intersec = arena_poly.intersection(circle)
                    arc_length = h_rest.intersection(h_intersec).length
                    winv_mat[i,j] = arc_length/(2*np.pi*dij)
                    #h_area = arena_poly.intersection(circle).area
                    #winv_mat[i,j]=h_area/(np.pi*dij**2)
    
    r = np.arange(0,60,0.2)
    K = np.zeros(len(r))
        
    for i in range(len(r)) :
        I_mat = np.zeros((exprnum, exprnum))
        I_mat[d_mat<=r[i]] = 1
        mul_w_I = np.multiply(winv_mat, I_mat)
        np.fill_diagonal(mul_w_I,0)
        K_d = mul_w_I.sum() / lmbda / exprnum
        K[i] = K_d
    
    return r, K
