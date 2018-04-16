import numpy as np

class rigid_registration(object):
  def __init__(self, X, Y, R=None, t=None, s=None, sigma2=None, maxIterations=100, tolerance=0.001, w=0):
    if X.shape[1] != Y.shape[1]:
      raise 'Both point clouds must have the same number of dimensions!'

    self.X             = X
    self.Y             = Y
    self.TY            = Y
    (self.N, self.D)   = self.X.shape
    (self.M, _)        = self.Y.shape
    self.R             = np.eye(self.D) if R is None else R
    self.t             = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
    self.s             = 1 if s is None else s
    self.sigma2        = sigma2
    self.iteration     = 0
    self.maxIterations = maxIterations
    self.tolerance     = tolerance
    self.w             = w
    self.q             = 0
    self.err           = 0

  def register(self, callback):
    self.initialize()

    while self.iteration < self.maxIterations and self.err > self.tolerance:
      self.iterate()
      if callback:
        callback(iteration=self.iteration, error=self.err, X=self.X, Y=self.TY)

    return self.TY, self.R, np.atleast_2d(self.t), self.s

  def iterate(self):
    self.EStep()
    self.MStep()
    self.iteration += 1

  def MStep(self):
    self.updateTransform()
    self.transformPointCloud()
    self.updateVariance()

  def updateTransform(self):
    muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
    muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

    self.XX = self.X - np.tile(muX, (self.N, 1))
    YY      = self.Y - np.tile(muY, (self.M, 1))

    self.A = np.dot(np.transpose(self.XX), np.transpose(self.P))
    self.A = np.dot(self.A, YY)

    U, _, V = np.linalg.svd(self.A, full_matrices=True)
    C = np.ones((self.D, ))
    C[self.D-1] = np.linalg.det(np.dot(U, V))

    self.R = np.dot(np.dot(U, np.diag(C)), V)

    self.YPY = np.dot(np.transpose(self.P1), np.sum(np.multiply(YY, YY), axis=1))

    self.s = np.trace(np.dot(np.transpose(self.A), self.R)) / self.YPY

    self.t = np.transpose(muX) - self.s * np.dot(self.R, np.transpose(muY))

  def transformPointCloud(self, Y=None):
    if Y is None:
      self.TY = self.s * np.dot(self.Y, np.transpose(self.R)) + np.tile(np.transpose(self.t), (self.M, 1))
      return
    else:
      return self.s * np.dot(Y, np.transpose(self.R)) + np.tile(np.transpose(self.t), (Y.shape[0], 1))

  def updateVariance(self):
    qprev = self.q

    trAR     = np.trace(np.dot(self.A, np.transpose(self.R)))
    xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.XX, self.XX), axis =1))
    self.q   = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
    self.err = np.abs(self.q - qprev)

    self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)

    if self.sigma2 <= 0:
      self.sigma2 = self.tolerance / 10

  def initialize(self):
    self.TY = self.s * np.dot(self.Y, np.transpose(self.R)) + np.repeat(self.t, self.M, axis=0)
    if not self.sigma2:
      XX = np.reshape(self.X, (1, self.N, self.D))
      YY = np.reshape(self.TY, (self.M, 1, self.D))
      XX = np.tile(XX, (self.M, 1, 1))
      YY = np.tile(YY, (1, self.N, 1))
      diff = XX - YY
      err  = np.multiply(diff, diff)
      self.sigma2 = np.sum(err) / (self.D * self.M * self.N)

    self.err  = self.tolerance + 1
    self.q    = -self.err - self.N * self.D/2 * np.log(self.sigma2)

  def EStep(self):
    P = np.zeros((self.M, self.N))

    for i in range(0, self.M):
      diff     = self.X - np.tile(self.TY[i, :], (self.N, 1))
      diff    = np.multiply(diff, diff)
      P[i, :] = P[i, :] + np.sum(diff, axis=1)

    c = (2 * np.pi * self.sigma2) ** (self.D / 2)
    c = c * self.w / (1 - self.w)
    c = c * self.M / self.N

    P = np.exp(-P / (2 * self.sigma2))
    den = np.sum(P, axis=0)
    den = np.tile(den, (self.M, 1))
    den[den==0] = np.finfo(float).eps
    den += c

    self.P   = np.divide(P, den)
    self.Pt1 = np.sum(self.P, axis=0)
    self.P1  = np.sum(self.P, axis=1)
    self.Np  = np.sum(self.P1)


class affine_registration(object):
  def __init__(self, X, Y, B=None, t=None, sigma2=None, maxIterations=100, tolerance=0.001, w=0):
    if X.shape[1] != Y.shape[1]:
        raise 'Both point clouds must have the same number of dimensions!'

    self.X             = X
    self.Y             = Y
    self.TY            = Y
    (self.N, self.D)   = self.X.shape
    (self.M, _)        = self.Y.shape
    self.B             = np.eye(self.D) if B is None else B
    self.t             = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
    self.sigma2        = sigma2
    self.iteration     = 0
    self.maxIterations = maxIterations
    self.tolerance     = tolerance
    self.w             = w
    self.q             = 0
    self.err           = 0

  def register(self, callback):
    self.initialize()

    while self.iteration < self.maxIterations and self.err > self.tolerance:
      self.iterate()
      if callback:
        callback(iteration=self.iteration, error=self.err, X=self.X, Y=self.TY)

    return self.TY, self.B, np.atleast_2d(self.t)

  def iterate(self):
    self.eStep()
    self.mStep()
    self.iteration += 1

  def mStep(self):
    self.updateTransform()
    self.transformPointCloud()
    self.updateVariance()

  def updateTransform(self):
    muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
    muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

    self.XX = self.X - np.tile(muX, (self.N, 1))
    YY      = self.Y - np.tile(muY, (self.M, 1))

    self.A = np.dot(np.transpose(self.XX), np.transpose(self.P))
    self.A = np.dot(self.A, YY)

    self.YPY = np.dot(np.transpose(YY), np.diag(self.P1))
    self.YPY = np.dot(self.YPY, YY)

    Bt = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
    self.B = np.transpose(Bt)
    self.t = np.transpose(muX) - np.dot(self.B, np.transpose(muY))

  def transformPointCloud(self, Y=None):
    if Y is None:
      self.TY = np.dot(self.Y, np.transpose(self.B)) + np.tile(np.transpose(self.t), (self.M, 1))
      return
    else:
      return np.dot(Y, np.transpose(self.B)) + np.tile(np.transpose(self.t), (Y.shape[0], 1))

  def updateVariance(self):
    qprev = self.q

    trAB     = np.trace(np.dot(self.A, np.transpose(self.B)))
    xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.XX, self.XX), axis =1))
    trBYPYP  = np.trace(np.dot(np.dot(self.B, self.YPY), np.transpose(self.B)))
    self.q   = (xPx - 2 * trAB + trBYPYP) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
    self.err = np.abs(self.q - qprev)

    self.sigma2 = (xPx - trAB) / (self.Np * self.D)

    if self.sigma2 <= 0:
      self.sigma2 = self.tolerance / 10

  def initialize(self):
    self.TY = np.dot(self.Y, np.transpose(self.B)) + np.repeat(self.t, self.M, axis=0)
    if not self.sigma2:
      XX = np.reshape(self.X, (1, self.N, self.D))
      YY = np.reshape(self.TY, (self.M, 1, self.D))
      XX = np.tile(XX, (self.M, 1, 1))
      YY = np.tile(YY, (1, self.N, 1))
      diff = XX - YY
      err  = np.multiply(diff, diff)
      self.sigma2 = np.sum(err) / (self.D * self.M * self.N)

    self.err  = self.tolerance + 1
    self.q    = -self.err - self.N * self.D/2 * np.log(self.sigma2)

  def eStep(self):
    P = np.zeros((self.M, self.N))

    for i in range(0, self.M):
      diff     = self.X - np.tile(self.TY[i, :], (self.N, 1))
      diff    = np.multiply(diff, diff)
      P[i, :] = P[i, :] + np.sum(diff, axis=1)

    c = (2 * np.pi * self.sigma2) ** (self.D / 2)
    c = c * self.w / (1 - self.w)
    c = c * self.M / self.N

    P = np.exp(-P / (2 * self.sigma2))
    den = np.sum(P, axis=0)
    den = np.tile(den, (self.M, 1))
    den[den==0] = np.finfo(float).eps

    self.P   = np.divide(P, den)
    self.Pt1 = np.sum(self.P, axis=0)
    self.P1  = np.sum(self.P, axis=1)
    self.Np  = np.sum(self.P1)


class deformable_registration(object):
  def __init__(self, X, Y, alpha=None, beta=None, sigma2=None, maxIterations=100, tolerance=0.001, w=0):
    if X.shape[1] != Y.shape[1]:
        raise 'Both point clouds must have the same number of dimensions!'

    self.X             = X
    self.Y             = Y
    self.TY            = Y
    (self.N, self.D)   = self.X.shape
    (self.M, _)        = self.Y.shape
    self.alpha         = 2 if alpha is None else alpha
    self.beta          = 2 if alpha is None else beta
    self.W             = np.zeros((self.M, self.D))
    self.G             = np.zeros((self.M, self.M))
    self.sigma2        = sigma2
    self.iteration     = 0
    self.maxIterations = maxIterations
    self.tolerance     = tolerance
    self.w             = w
    self.q             = 0
    self.err           = 0

  def register(self, callback):
    self.initialize()

    while self.iteration < self.maxIterations and self.err > self.tolerance:
        self.iterate()
        if callback:
            callback(iteration=self.iteration, error=self.err, X=self.X, Y=self.TY)

    return self.TY, np.dot(self.G, self.W)

  def iterate(self):
    self.EStep()
    self.MStep()
    self.iteration += 1

  def MStep(self):
    self.updateTransform()
    self.transformPointCloud()
    self.updateVariance()

  def updateTransform(self):
    A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
    B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
    self.W = np.linalg.solve(A, B)

  def transformPointCloud(self, Y=None):
    if Y is None:
      self.TY = self.Y + np.dot(self.G, self.W)
      return
    else:
      return Y + np.dot(self.G, self.W)

  def updateVariance(self):
    qprev = self.sigma2

    xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
    yPy      = np.dot(np.transpose(self.P1),  np.sum(np.multiply(self.Y, self.Y), axis=1))
    trPXY    = np.sum(np.multiply(self.Y, np.dot(self.P, self.X)))

    self.sigma2 = (xPx - trPXY) / (self.Np * self.D)

    if self.sigma2 <= 0:
      self.sigma2 = self.tolerance / 10

    self.err = np.abs(self.sigma2 - qprev)

  def initialize(self):
    if not self.sigma2:
      XX = np.reshape(self.X, (1, self.N, self.D))
      YY = np.reshape(self.Y, (self.M, 1, self.D))
      XX = np.tile(XX, (self.M, 1, 1))
      YY = np.tile(YY, (1, self.N, 1))
      diff = XX - YY
      err  = np.multiply(diff, diff)
      self.sigma2 = np.sum(err) / (self.D * self.M * self.N)

    self.err  = self.tolerance + 1
    self.q    = -self.err - self.N * self.D/2 * np.log(self.sigma2)
    self._makeKernel()

  def EStep(self):
    P = np.zeros((self.M, self.N))

    for i in range(0, self.M):
      diff     = self.X - np.tile(self.TY[i, :], (self.N, 1))
      diff    = np.multiply(diff, diff)
      P[i, :] = P[i, :] + np.sum(diff, axis=1)

    c = (2 * np.pi * self.sigma2) ** (self.D / 2)
    c = c * self.w / (1 - self.w)
    c = c * self.M / self.N

    P = np.exp(-P / (2 * self.sigma2))
    den = np.sum(P, axis=0)
    den = np.tile(den, (self.M, 1))
    den[den==0] = np.finfo(float).eps
    den += c

    self.P   = np.divide(P, den)
    self.Pt1 = np.sum(self.P, axis=0)
    self.P1  = np.sum(self.P, axis=1)
    self.Np  = np.sum(self.P1)

  def _makeKernel(self):
    XX = np.reshape(self.Y, (1, self.M, self.D))
    YY = np.reshape(self.Y, (self.M, 1, self.D))
    XX = np.tile(XX, (self.M, 1, 1))
    YY = np.tile(YY, (1, self.M, 1))
    diff = XX-YY
    diff = np.multiply(diff, diff)
    diff = np.sum(diff, 2)
    self.G = np.exp(-diff / (2 * self.beta))

from functools import reduce

class jrmpc_rigid(object):
  def __init__(self, Y, R=None, t=None, maxIterations=100, gamma=0.1, ):
    if Y is None:
      raise 'Empty list of point clouds!'

    dimensions = [cloud.shape[1] for cloud in Y]

    if not all(dimension == dimensions[0] for dimension in dimensions):
      raise 'All point clouds must have the same number of dimensions!'

    self.Y = Y
    self.M = [cloud.shape[0] for cloud in self.Y]
    self.D = dimensions[0]

    if R:
      rotations = [rotation.shape for rotation in R]
      if not all(rotation[0] == self.D and rotation[1] == self.D for rotation in rotations):
        raise 'All rotation matrices need to be %d x %d matrices!' % (self.D, self.D)
      self.R = R
    else:
      self.R = [np.eye(self.D) for cloud in Y]

    if t:
      translations = [translations.shape for translation in t]
      if not all(translations[0] == 1 and translations[1] == self.D for translation in translations):
        raise 'All translation vectors need to be 1 x %d matrices!' % (self.D)
      self.t = t
    else:
      self.t = [np.atleast_2d(np.zeros((1, self.D))) for cloud in self.Y]

  def initializeGMMMeans(self):
    self.K = np.median(self.M)
    az = 2 * np.pi * np.random.rand(self.K)
    el = 2 * np.pi * np.random.rand(self.K)
    self.X = np.array([np.multiply(np.cos(az), np.cos(el)), np.sin(el), np.multiply(np.sin(az), np.cos(el))])

  def print_self(self):
    print('Y has %d point clouds.' % (len(self.Y)))
    print('Each point cloud has M points: ', self.M)
    print('Dimensionality of all point clouds is: ', self.D)
    print('Rotation matrices are: ', self.R)
    print('Translation vectors are: ', self.t)
