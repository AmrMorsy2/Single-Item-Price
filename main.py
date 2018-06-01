import numpy as np
import statsmodels.api as sm

#Generate random training data
def generate_data(sz,rows):
    #Random number of items
    data = np.random.randint(10, size=(rows,sz))
    #Original Price
    orginal = np.random.randint(100,size=sz)
    data = sm.add_constant(data)  # Adding bias input
    sz = sz+1
    h , w = data.shape
    ret = np.array([[]],dtype=np.object)
    d = np.zeros(rows)
    #Caluclating actual cost
    for i in range(h):
        sum=0
        for j in range(1,sz):
            sum += data[i][j]*orginal[j-1]
        tmp = np.array([[],np.int],dtype=np.object)
        if i==0:
            ret = data[i]
        else:
            ret = np.vstack((ret,data[i]))
        d[i] = sum
    data = ret
    return data,d,orginal

def train(dim=5,epoch=1000,eta=0.001):
    #Fetch training set
    data , d, original = generate_data(dim,50)
    weight = np.random.rand(len(data[0]))
    weight *= 1000
    for i in range(epoch):
        #run on all data sample
        for j in range(len(data)):
            v = np.dot(data[j], weight)
            actual = v;
            desired = d[j]
            error= desired - actual
            rate = eta * error
            tmp = np.array(data[j],dtype=np.float64)
            weight += (rate * tmp)
    return weight,original


def test(dim=5):
    weight,original = train(dim)
    #Randomize a test sample
    sample = np.append(np.array(1),np.random.randint(10, size=(1,dim)))
    print(sample)
    desired = np.dot(sample, weight)
    actual = np.dot(sample[1:len(sample)], original)
    print("Actual = "+str(actual))
    print("Desired = "+str(desired))
    print("Absolute error = "+str(abs(desired-actual)))
    print("Error % = "+str(abs(desired-actual)/actual))

test()