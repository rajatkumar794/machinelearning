import numpy as np

def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

def sigmoidGradient(z):
    g=sigmoid(z)
    g=g*(1-g)
    return g


def initializeWeights(L_in, L_out):
    W=np.zeros((L_out, 1+L_in))
    epsilon=0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon - epsilon
    return W


def Costnn(theta, X, y, ilayer, hlayer, olayer, lambd):
    theta1 = np.reshape(theta[0:(hlayer * (ilayer + 1))], (hlayer, ilayer + 1))
    theta2 = np.reshape(theta[0:(olayer * (hlayer + 1))], (olayer, hlayer + 1))
    m = np.size(X, 0)
    y_vec = np.zeros((m, olayer));
    for p in range(m):
        t = int(y[p])
        y_vec[p, t] = 1

    J = 0
    X = np.c_[(np.ones((m, 1)), X)]

    z = np.dot(theta1, X.transpose())
    a1 = sigmoid(z)
    a1 = a1.transpose()
    a1 = np.c_[(np.ones((m, 1)), a1)]
    z = np.dot(theta2, a1.transpose())
    a2 = sigmoid(z)

    J = np.multiply(np.log(a2), y_vec.transpose()) + np.multiply(np.log(1 - a2), (1 - y_vec).transpose())
    J = np.sum(J)
    t1 = np.square(theta1)
    t2 = np.square(theta2)
    J = J / (-m) + (lambd / (2 * m)) * (np.sum(t1[:, 1:]) + np.sum(t2[:, 1:]))

    return J


def grad(theta, X, y, ilayer, hlayer, olayer, lambd):
    theta1 = np.reshape(theta[0:(hlayer * (ilayer + 1))], (hlayer, ilayer + 1))
    theta2 = np.reshape(theta[0:(olayer * (hlayer + 1))], (olayer, hlayer + 1))
    m = np.size(X, 0)
    y_vec = np.zeros((m, olayer));
    for p in range(m):
        t = int(y[p])
        y_vec[p, t] = 1

    X = np.c_[(np.ones((m, 1)), X)]
    theta1_grad = np.zeros(np.shape(theta1))
    theta2_grad = np.zeros(np.shape(theta2))

    z = np.dot(theta1, X.transpose())
    a1 = sigmoid(z)
    a1 = a1.transpose()
    a1 = np.c_[(np.ones((m, 1)), a1)]
    z = np.dot(theta2, a1.transpose())
    a2 = sigmoid(z)

    d2 = a2 - y_vec.transpose()
    t = sigmoidGradient(np.dot(theta1, X.transpose()))
    d1 = np.dot(d2.transpose(), theta2[:, 1:])
    d1 = np.multiply(d1, t.transpose())
    del1 = np.dot(d1.transpose(), X)
    del2 = np.dot(d2, a1)
    theta1_grad[:, 0] = (1 / m) * del1[:, 0];
    theta2_grad[:, 0] = (1 / m) * del2[:, 0];
    theta1_grad[:, 1:] = (1 / m) * del1[:, 1:] + (lambd / m) * theta1[:, 1:];
    theta2_grad[:, 1:] = (1 / m) * del2[:, 1:] + (lambd / m) * theta2[:, 1:];

    theta1 = theta1_grad.reshape(((ilayer + 1) * hlayer, 1))
    theta2 = theta2_grad.reshape(((hlayer + 1) * olayer, 1))
    th = np.r_[(theta1, theta2)]
    #    th=th.flat
    #    param=[]
    #    for i in th:
    #        param.append(i)
    #    return np.array(param)
    return th


def predict(Theta1, Theta2, X):
    m = np.size(X, 0)
    X = np.c_[(np.ones((m, 1)), X)]

    z = np.dot(X, Theta1.transpose())
    h1 = sigmoid(z);
    h1 = np.c_[(np.ones((m, 1)), h1)]
    z = np.dot(h1, Theta2.transpose())
    h2 = sigmoid(z);
    pred = np.argmax(h2, 1)
    return pred



train=np.genfromtxt('mnist_train.csv', delimiter=',')
tt=np.genfromtxt('test.csv', delimiter=',')
tt=tt[1:,:]

ilayer=784
olayer=10
hlayer=570

X_train=train[:,1:]
y_train=train[:,0]

Theta1=initializeWeights(ilayer, hlayer)
Theta2=initializeWeights(hlayer, olayer)

theta1=Theta1.reshape(((ilayer+1)*hlayer,1))
theta2=Theta2.reshape(((hlayer+1)*olayer,1))
initial_param=np.r_[(theta1,theta2)]

#args = (X_train, y_train, ilayer, hlayer, olayer, 1)  # parameter values

for i in range(0,400):
    param=grad(initial_param,X_train, y_train, ilayer, hlayer, olayer,1)
    initial_param=initial_param-1*param
J=Costnn(initial_param,X_train, y_train, ilayer, hlayer, olayer,1)

theta1=np.reshape(initial_param[0:(hlayer*(ilayer+1))], (hlayer, ilayer+1))
theta2=np.reshape(initial_param[0:(olayer*(hlayer+1))], (olayer, hlayer+1))
pred=predict(theta1, theta2, tt)
#temp=map(int, (pred==y_train))
#res=[]
#for i in temp:
#    res.append(i)
#acc=np.mean(res)*100
print(J)
#print(acc)