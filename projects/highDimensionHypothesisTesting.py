from Ball import bcov_test,bcov,bcorsis,bd_test
import numpy as np
def argCheck(list_of_lists):
    if type(list_of_lists) is not list:
        return "sample is not list!"
    if len(list_of_lists)<1:
        return "no samples are present!"
    x = []
    for listt in list_of_lists:
        if(len(listt)==0):
            return "sample with 0 data is present!"
        x.append(len(listt))
    lx = len(list_of_lists[0])
    for y in list_of_lists:
        if lx!=len(y):
            return "equal lengths samples are not present!"
    return True

def highdimension_test(list_of_lists):
    check = argCheck(list_of_lists)
    if check!=True:
        return check
    results = bcov_test(list_of_lists)
    return results

def highdimension_covariance(list_of_lists):
    check = argCheck(list_of_lists)
    if check!=True:
        return check
    results = bcov(list_of_lists)
    return results

def argCheckRT(n,p,y,params,d,method):
    if method not in ["lm","gam"]:
        return "method not in lm and gam!"
    if type(n)!=int:
        return "n is not integer"
    if type(p)!=int:
        return "p is not integer"
    if n<=0:
        return "n<=0"
    if p<=0:
        return "p<=0"
    if not isinstance(y, np.ndarray):
        return "y is not list"
    if type(params)!=list:
        return "params is not list"
    return True  
def reproduce_target(n,p,y,params,d,method = "lm"):
    import numpy as np
    check = argCheckRT(n,p,y,params,d,method)
    if check!=True:
        return check
    mean = np.zeros(p)
    cov = np.array([0.5]*np.square(p)).reshape((p, p))
    cov[np.diag_indices(p)] = 1
    x = np.random.multivariate_normal(mean,cov,n)
    error = np.random.normal(0,1,n)
    x_num = np.ones(p)
   
    # print("here!")
    if method not in ["lm","gam"]:
        targets = bcorsis(y,x,x_num,method=method,d = d)
    else:
        targets = bcorsis(y,x,x_num,method=method,params = params,d = d)
    # print(targets)
    return targets

def argCheck1D(x,y):
    import numpy as np
    if not isinstance(x,np.ndarray):
        return "x is not list!"
    if not isinstance(y,np.ndarray):
        return "y is not list!"
    if x.shape!=y.shape:
        return "shapes are not equal!"
    return True
def argCheckND(x,y):
    import numpy as np
    if not isinstance(x,np.ndarray):
        return "x is not list!"
    if not isinstance(y,np.ndarray):
        return "y is not list!"
    if x.shape!=y.shape:
        return "shapes are not equal!"
    return True
def argCheckDSTMAT(x,y):
    import numpy as np
    if not isinstance(x,np.ndarray):
        return "x is not list!"
    if not isinstance(y,np.ndarray):
        return "y is not list!"
    if x.shape!=y.shape:
        return "shapes are not equal!"
    return True
def argCheckAllTypeTest(x,y,type,size):
    import numpy as np
    if not isinstance(x,np.ndarray):
        return "x is not list!"
    if not isinstance(y,np.ndarray):
        return "y is not list!"
    if type not in ["1D","ND","DistanceMatrix"]:
        return "type is not one of 1D,ND,DistanceMatrix!"
    if not isinstance(size,int):
        return "type of size is not integer"
    return True

def AllType_test(x,y,type,size = 50):
    import numpy as np
    check = argCheckAllTypeTest(x,y,type,size)
    if check!=True:
        return check
    if type=="1D":
        check = argCheck1D(x,y)
        if check!=True:
            return check
        return bd_test(x,y)
    elif type=="ND":
        check = argCheckND(x,y)
        if check!=True:
            return check
        return bd_test(x,y)
    elif type=="DistanceMatrix":
        check = argCheckDSTMAT(x,y)
        if check!=True:
            return check
        from sklearn.metrics.pairwise import euclidean_distances
        x = np.row_stack((x, y))
        dx = euclidean_distances(x, x)
        data_size = [size, size]
        return bd_test(dx, size=data_size, dst=True) 

if __name__=="__main__":
    list_of_lists = [[1],[2]]
    print(highdimension_test(list_of_lists))

    list_of_lists = [[1],[2]]
    print(highdimension_covariance(list_of_lists))

    n = 150
    p = 3000
    mean = np.zeros(p)
    cov = np.array([0.5]*np.square(p)).reshape((p, p))
    cov[np.diag_indices(p)] = 1
    x = np.random.multivariate_normal(mean,cov,n)
    error = np.random.normal(0,1,n)
    y = 4*np.square(x[:, 2])+6*np.square(x[:, 1])+8*x[:, 3]-10*x[:,4]+error 
    numTargets = 11
    print(reproduce_target(n,p,y,[5,3],numTargets))

    x = np.random.normal(0, 1, 50)
    y = np.random.normal(1, 1, 50)
    type = "1D"
    size = 50
    print(AllType_test(x,y,type,size = size))

    x = np.random.normal(0, 1, 100).reshape(50, 2)
    y = np.random.normal(3, 1, 100).reshape(50, 2)
    type = "ND"
    size = 50
    print(AllType_test(x,y,type,size = size))

    sigma = [[1, 0], [0, 1]]
    x = np.random.multivariate_normal(mean=[0, 0], cov=sigma, size=size)
    y = np.random.multivariate_normal(mean=[1, 1], cov=sigma, size=size)        
    type = "DistanceMatrix"
    size = 50
    print(AllType_test(x,y,type,size = size))  
