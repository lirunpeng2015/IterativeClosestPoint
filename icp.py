import numpy as np

def icp(reference: np.array, source: np.array, max_iterations=100, tolerance=1e-6):
    """Performs ICP and returns rotation and translation. 
        Expected inputs in the form [x y]
                                    [x y]
                                    [x y]
                                    [x y]"""
    
    # check for proper shape of point clouds
    if np.shape(reference)[1] != 1 & len(np.shape(reference)) != 2:
        print("Error: reference shape is not correct")
        return
    
    if np.shape(source)[1] != 1 & len(np.shape(source)) != 2:
        print("Error: source shape is not correct")
        return
    
    # initialize rotation matrix as identity matrix
    # initialize translation as 1x2 array of zeros
    R = np.eye(2)
    t = np.zeros(1,2)
    
    
    return R, t