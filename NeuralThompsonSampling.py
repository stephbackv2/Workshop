import numpy as np
import numpy.random as rd

def sampleInitParams(dist, inputDim, width, numLayers, specialRule=False):
    """
    Used to sample params for a fully connected neural net from the params
    :param dist:
    :param inputDim:
    :param width:
    :return: sample from distribution as dict
    """
    # they used some more structured matrices as initial stuff
    if specialRule:
        sizeFl = (inputDim//2, width//2)
        firstLayer = dist(0, 4/width, sizeFl)
        flZeros = np.zeros(sizeFl)
        sizeMl = (numLayers, width//2, width//2)
        midLayer = dist(0, 4/width, sizeMl)
        mlZeros = np.zeros(sizeMl)
        lastLayer = dist(0, 4/width, width//2)
        if width % 2 == 0:
            lastLayer = np.concatenate((lastLayer, -lastLayer)).reshape(width,1)
        else:
            lastLayer = np.concatenate((lastLayer, dist(0, 2/width, 1), -lastLayer)).reshape(width,1)
        dictTmp = {
            "firstLayer": np.concatenate((
                np.concatenate((firstLayer, flZeros), axis=1),
                np.concatenate((flZeros, firstLayer), axis=1)
            )),
            "midLayers": np.concatenate((
                np.concatenate((midLayer, mlZeros), axis=2),
                np.concatenate((mlZeros, midLayer), axis=2)
            ), axis=1),
            "lastLayer": lastLayer
        }
    else:
        dictTmp = {
            "firstLayer": dist(0, 4/width, (inputDim, width)),
            "midLayers": dist(0, 4/width, (numLayers, width, width)), #they used some more structured matrices as initial stuff
            "lastLayer": dist(0, 2/width, (width, 1))
        }
    return dictTmp
class NetworkLayer(object):
    def __init__(self, inputDim, outputDim, matrix, vector=None, scalar=1):
        if (matrix.shape == (inputDim, outputDim)):
            self.matrix = matrix
        else:
            print(f"Dimensions don't match. Weight matrix should be of shape {(inputDim, outputDim)}, but is of shape {matrix.shape}")
            self.matrix = rd.normal(0, 1, (inputDim, outputDim))
        if not vector is None:
            self.vector = vector
            self.bias = True
        else:
            self.bias = False
            self.vector = np.zeros(outputDim)
        self.inputDim = inputDim
        self.scalar = scalar
        self.weightGrad = np.eye(inputDim, outputDim)
        self.cache = (np.zeros(inputDim), np.zeros(inputDim))

    def forward(self, input):
        #print(input.shape, self.matrix.shape)
        input = input.reshape(1, self.matrix.shape[0])
        a = input @ self.matrix + self.vector
        if self.scalar == 1:
            out = np.maximum(0, a)
        else:
            out = self.scalar * a
        #for backwards step
        self.cache = (input, a)
        return out

    def backward(self, dout, lr=None):
        #print(f"outputErr: {dout}")
        #print(f"weights: {self.matrix}")
        fc_cache, relu_cache = self.cache
        #print(f"cache: {fc_cache}")
        #relu Backward
        x = relu_cache
        if self.scalar == 1:
            da = dout * np.where(x > 0, 1, 0)
        else:
            da = dout * self.scalar
        #print(f"da: {da}")
        #affine Backward
        x = fc_cache
        dout = da
        db = dout
        dx = dout @ self.matrix.T
        dw = x.T @ dout
        self.weightGrad = dw
        #print(f"in_error {dx}")
        #print(f"weights_error {dw}")
        if not lr is None:
            self.updateParams(self.matrix - lr*dw, self.vector - lr*db)
        return dx

    def updateParams(self, matrix, vector):
        self.matrix = matrix
        if self.bias:
            self.vector = vector
class FullyConnectedNeuralNetwork(object):
    def __init__(self, inputDim, width=10, numMidLayers=3, bias=False, loss=None, lossPrime=None, specialRule=False):
        self.inputDim = inputDim
        self.width = width
        self.numLayers = numMidLayers
        # set loss
        self.loss = loss
        self.loss_prime = lossPrime
        # draw some initial params
        params = sampleInitParams(rd.normal, inputDim, width, numMidLayers, specialRule)
        # create our layers
        layers = list()
        if bias:
            layers.append(NetworkLayer(inputDim, width, params["firstLayer"], np.zeros(width)))
            for i in range(numMidLayers):
                layers.append(NetworkLayer(width, width, params["midLayers"][i, :, :], np.zeros(width)))
            layers.append(NetworkLayer(width, 1, params["lastLayer"], np.zeros(1)))
        else:
            layers.append(NetworkLayer(inputDim, width, params["firstLayer"]))
            for i in range(numMidLayers):
                layers.append(NetworkLayer(width, width, params["midLayers"][i, :, :]))
            layers.append(NetworkLayer(width, 1, params["lastLayer"]))
        self.layers = layers
        self.params = params
        self.initLayers = layers
        self.history = np.array([], dtype=np.dtype("uint16")).reshape(0, 2)  # action, reward list
        self.arms = np.eye(inputDim, inputDim).reshape(inputDim, 1, inputDim)
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def updateLoss(self, loss, lossPrime):
        self.loss = loss
        self.loss_prime = lossPrime
    def updateParams(self, x, y, lr=1e-2):
        output = self.predict(x)
        #print([(x.matrix, x.vector) for x in self.layers])
        err = self.loss(y, output)
        error = self.loss_prime(y, output)
        #print(f"err: {error}\n y:{y}\n output: {output}")
        for layer in reversed(self.layers):
            error = layer.backward(error, lr)
        return err


def layersToVector(layers, netParams):
    grad = np.zeros(netParams)
    index = 0
    for layer in layers:
        numParams = np.prod(layer.matrix.shape)
        # flatten to make it a vector
        matrix = layer.matrix.flatten()
        grad[index:index + numParams] = matrix
        index += numParams
    return grad
class ThompsonNetwork(FullyConnectedNeuralNetwork):
    def __init__(self, inputDim, width=10, numMidLayers=3, specialRule=True, lamb=1, nu=1):
        # reg Param
        self.lamb = lamb
        # exploration variance
        self.nu = nu
        numParams = inputDim*width + numMidLayers * width**2 + width
        self.numParams = numParams
        self.U = lamb * np.eye(numParams, numParams)
        super().__init__(inputDim, width, numMidLayers, False, None, None, specialRule)
    def tsLoss(self):
        sum = 0
        for action, reward in self.history:
            sum += (self.predict(action)-reward)**2

        params = self.layers
        theta0 = layersToVector(self.initLayers, self.numParams)
        theta = layersToVector(params, self.numParams)
        return sum/2 + self.width * self.lamb * np.linalg.norm(theta-theta0)**2
    def tsGrad(self, x):
        """
        Gradient of Neural Network in numParamsx1 shape
        :param x:
        :return:
        """
        grad = np.zeros(self.numParams)
        index = self.numParams
        output = self.predict(x)
        #careful, we are going backwards
        for layer in reversed(self.layers):
            numParams = np.prod(layer.matrix.shape)
            layer.backward(1)
            #flatten to make it a vector
            layerGrad = layer.weightGrad.flatten()
            grad[index-numParams:index] = layerGrad
            index -= numParams
        return grad
    def calcSigma(self, x):
        g = self.tsGrad(x)
        #its quiet big and stuff, so only diag elements
        pseudoInvU = np.diag(1/np.diagonal(self.U))
        return self.lamb/self.width * g.T @ pseudoInvU @ g

    def sampleReward(self, x):
        """
        Want to do that for each arm
        :param x:
        :return:
        """
        return rd.normal(self.predict(x), self.calcSigma(x)*self.nu**2)

    def updateParamsFromVec(self, vec):
        index = 0
        for layer in self.layers:
            shape = layer.matrix.shape
            numPar = np.prod(shape)
            layer.matrix = vec[0, index:index+numPar].reshape(shape)
            index += numPar
    def updateTheta(self, steps, lr=1e-3):
        """
        solve gradient descent for steps number of steps
        :param steps:
        :return:
        """
        for step in range(steps):
            #calc grad of loss function
            firstSum = np.sum(
                np.apply_along_axis(
                    lambda x: self.tsGrad(self.arms[x[0]]) * (self.predict(self.arms[x[0]]) - x[1]),
                    1,
                    self.history
                ), 0)  # axis
            paramVec = layersToVector(self.layers, self.numParams)
            diff = paramVec-layersToVector(self.initLayers, self.numParams)
            secondSum = self.width*self.lamb*diff
            direction = firstSum + secondSum
            newParamVec = paramVec - lr*direction
            self.updateParamsFromVec(newParamVec)

    def updateU(self, xPlayed):
        g = self.tsGrad(xPlayed)
        self.U = self.U + g @ g.T / self.width

    #not used atm
    def play(self, env, arms):
        """
        playing one round of everything and update all
        :param arm:
        :return:
        """
        predicts = np.apply_along_axis(net.predict, 0, arms).reshape(len(arms))
        armsPlayed = (-predicts).argsort()[:env.num_positions]
        rew = env.get_stochastic_reward(armsPlayed)
        obs = env.get_observation()
        #ToDo append along the right thingy
        for fail in obs["round_failure"]:
            self.history.append([fail, 0])
        for success in obs["round_success"]:
            self.history.append([fail, 0])
        #update Theta
        self.updateTheta(100, 1e-3)
        #updateU for each arm played
        for arm in arms[armsPlayed]:
            self.updateU(arm)

if __name__=="__main__":
    # loss function and its derivative
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2));


    def msePrime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size;

    rd.seed(3)
    #test on xor
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    net = FullyConnectedNeuralNetwork(2, 7, 4, False, mse, msePrime)
    samples = len(x_train)
    for i in range(1000):
        err = 0
        for j in range(samples):
            err += net.updateParams(x_train[j], y_train[j], 0.1)

        # calculate average error on all samples
        err /= samples
        print('epoch %d/%d   error=%f' % (i + 1, 1000, err))


