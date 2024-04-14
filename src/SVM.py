import numpy as np
from tqdm import tqdm
from typing import Tuple

from kernels import GaussianKernel,LinearKernel,LaplacianRBFKernel,HellingerKernel,SublinearRBFKernel,GaussianKernel_orientation

class SVM_perso:
    '''
    Personal Implementation of the SMO method
    '''
    def __init__(self,C, kernel,epsilon, MaxLoop = 1e3, ShowLoop = False):
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.tol = epsilon # = eps ?
        self.MaxLoop = MaxLoop
        self.ShowLoop = ShowLoop
        
        self.alpha = None
        self.X = None
        self.y = None
        self.N = None
        self.error_list = None
        
        self.Loop = 0

    def evaluate(self,x):
        return np.sum([self.y[i] * self.alpha[i] * self.kernel(self.X[i],x) for i in range(self.N)]) - self.b

    def error(self,i):
        return self.evaluate(self.X[i]) - self.y[i]

    def takeStep(self,i1,i2):
        
        if i1 == i2:
            return 0

        y1, y2 = self.y[i1], self.y[i2]
        alph1, alph2 = self.alpha[i1], self.alpha[i2]
        E1, E2 = self.error_list[i1],self.error_list[i2]
        s = y1 * y2
        
        if y1 == y2:
            L = max(0, alph2 + alph1 - self.C)
            H = min(self.C,alph2 + alph1)
        else:
            L = max(0,alph2-alph1)
            H = min(self.C,self.C + alph2 - alph1)
        
        if L == H:
            return 0
    
        k11 = self.kernel(self.X[i1],self.X[i1])
        k12 = self.kernel(self.X[i1],self.X[i2])
        k22 = self.kernel(self.X[i2],self.X[i2])
        eta = k11+k22-2*k12
        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2)/eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        elif eta == 0:
            return 0
        else:
            print('Problem with the kernel')

        if abs(a2-alph2) < self.epsilon * (a2+alph2+self.epsilon):
            return 0

        a1 = alph1+s*(alph2-a2)

        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b
        
        if 0 < a1 < self.C:
            b_new = b1
        elif 0 < a2 < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2
        
        if 0 < a1 < self.C:
            self.error_list[i1] = 0
        if 0 < a2 < self.C:
            self.error_list[i2] = 0

        for i in range(self.N):
            if i!=i1 and i!=i2:
                self.error_list[i] = self.error_list[i] + y1 * (a1 - alph1) * self.kernel(self.X[i1], self.X[i]) + y2 * (a2 - alph2) * self.kernel(self.X[i2], self.X[i]) + self.b - b_new
        
        self.b = b_new

        self.alpha[i1] = a1
        self.alpha[i2] = a2
        
        return 1

    def examineExample(self,i2):
        y2 = self.y[i2]
        alph2 = self.alpha[i2]
        E2 = self.error_list[i2]
        r2 = y2 * E2
        if ((r2 < -self.tol) and (alph2 < self.C)) or ((r2 > self.tol) and (alph2 > 0)):
            N_in_interval = np.count_nonzero((self.alpha > 0) & (self.alpha < self.C))
            if N_in_interval > 1:
                i1 = np.argmax([np.abs(E2 - self.error(i1)) for i1 in range(self.N)])
                if self.takeStep(i1,i2):
                    return 1
            start = np.random.randint(0,self.N)
            for i in range(self.N):
                if (self.alpha[(i+start)%self.N] < self.C) and (self.alpha[(i+start)%self.N] > 0):
                    i1 = (i+start)%self.N
                    if self.takeStep(i1,i2):
                        return 1
            start = np.random.randint(0,self.N)
            for i in range(self.N):
                i1 = (i+start)%self.N
                if self.takeStep(i1,i2):
                    return 1
        return 0

    def fit(self,X,y):
        self.X = X
        self.y = y
        self.N = len(y)
        self.Loop = 0
        
        self.alpha = np.zeros(self.N)
        self.b = 0

        self.error_list = np.array([self.evaluate(X[i]) - y[i] for i in range(self.N)])

        numChanged = 0
        examineAll = 1
        while (numChanged>0 or examineAll) and (self.MaxLoop >= self.Loop):
            numChanged = 0
            if examineAll:
                for i in range(self.N):
                    numChanged += self.examineExample(i)
            else:
                for i in range(self.N):
                    if self.alpha[i] < self.C and self.alpha[i] > 0:
                        numChanged += self.examineExample(i)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
            if self.Loop % 10 and self.ShowLoop:
                print(self.Loop)
            self.Loop += 1

    def predict(self,x):
        N_pred = len(x) 
        pred = np.array([2 * (self.evaluate(x[i])>0) - 1 for i in range(N_pred)])
        return pred
    
class Solver:
    """
    Abstract class for solver. Override to implement a Solver.
    """

    def __init__(
        self
    ) -> None:

        pass

    def predict(
        self
    ) -> None:

        pass

    def fit(
        self
    ) -> None:

        pass


class SVM(Solver):
    
    """
    Implementation of SMO adapted to our kernels from https://github.com/itsikad/svm-smo
    """
    
    def __init__(
        self,
        c: float = 1.,
        kkt_thr: float = 1e-3,
        max_iter: int = 1e4,
        kernel_type: str = 'linear',
        gamma_rbf: float = 1.
    ) -> None:
        """
        Arguments:
            c: Penalty parameter, trade-offs wide margin (lower c) and small number of margin failures

            kkt_thr: threshold for satisfying the KKT conditions

            max_iter: maximal iteration for training

            kernel_type: can be either 'linear' or 'rbf'

            gamma: gamma factor for RBF kernel
        """

        if not kernel_type in ['linear', 'rbf']:
            raise ValueError('kernel_type must be either {} or {}'.format('linear', 'rbf'))

        super().__init__()

        # Initialize
        self.c = float(c)
        self.max_iter = max_iter
        self.kkt_thr = kkt_thr
        if kernel_type == 'linear':
            #self.kernel = self.linear_kernel
            self.kernel = LinearKernel()
        elif kernel_type == 'rbf':
            #self.kernel = self.rbf_kernel
            #self.gamma_rbf = gamma_rbf
            self.kernel = GaussianKernel(1/np.sqrt(2*gamma_rbf))

        self.b = np.array([])  # SVM's threshold
        self.alpha = np.array([])  # Alpha parameters of the support vectors
        self.support_vectors = np.array([])  # Matrix in which each row is a support vector
        self.support_labels = np.array([])  # Vector with the ground truth labels of the support vectors

    def predict(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        SVM predict method.

        Arguments:
            x : (N,D) data matrix, each row is a sample.

        Return:
            pred : SVM predictions, the i-th element corresponds to the i-th sample, y={-1,+1}

            scores: raw scores per sample
        """

        w = self.support_labels * self.alpha
        #x = self.kernel(self.support_vectors, x)
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]
        x = self.kernel.build_K(self.support_vectors, x,verbose=False)
        scores = np.matmul(w, x) + self.b
        pred = np.sign(scores)

        return pred, scores

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:

        """
        Train SVM classifier.

        Arguments:
            x_train : (N,D) data matrix, each row is a sample.

            y_train : Labels vector, y must be {-1,1}
        """

        # Initialize
        N, D = x_train.shape
        self.b = 0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train

        iter_idx = 0
        non_kkt_array = np.arange(N)  # Iterate all indices first
        error_cache = self.predict(x_train)[1] - y_train  # auxilary array for heuristics

        print("SVM training using SMO algorithm - START")
        while iter_idx < self.max_iter:

            # Pick samples
            i_2, non_kkt_array = self.i2_heuristic(non_kkt_array)
            if i_2 == -1:
                # All alphas satisfies KKT conditions
                break

            i_1 = self.i1_heuristic(i_2, error_cache)

            if i_1 == i_2:
                # Same sample, skip
                continue

            # Extract samples and labels
            x_1, y_1, alpha_1 = self.support_vectors[i_1, :], self.support_labels[i_1], self.alpha[i_1]
            x_2, y_2, alpha_2 = self.support_vectors[i_2, :], self.support_labels[i_2], self.alpha[i_2]

            # Update boundaries
            L, H = self.compute_boundaries(alpha_1, alpha_2, y_1, y_2)
            if L == H:
                continue

            # Compute eta
            eta = self.compute_eta(x_1, x_2)
            if eta == 0:
                continue

            # Calculate predictions and errors for x_1 and x_2
            _, score_1 = self.predict(x_1)
            _, score_2 = self.predict(x_2)

            E_1 = score_1 - y_1
            E_2 = score_2 - y_2

            # Compute and clip alpha_2
            alpha_2_new = alpha_2 + y_2 * (E_1 - E_2) / eta
            alpha_2_new = np.minimum(alpha_2_new, H)
            alpha_2_new = np.maximum(alpha_2_new, L)

            # Compute alpha_1
            alpha_1_new = alpha_1 + y_1*y_2*(alpha_2 - alpha_2_new)

            # Update threshold b (must be before updating alpha's)
            self.compute_b(alpha_1_new, alpha_2_new, E_1, E_2, i_1, i_2)

            # Update alpha vector
            self.alpha[i_1] = alpha_1_new
            self.alpha[i_2] = alpha_2_new

            # Update error cache
            error_cache[i_1] = self.predict(x_1)[1] - y_1
            error_cache[i_2] = self.predict(x_2)[1] - y_2

            iter_idx = iter_idx + 1

        # Store only support vectors
        support_vectors_idx = (self.alpha != 0)
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

        print(f"Training summary: {iter_idx} iterations, {self.alpha.shape[0]} supprts vectors")
        print("SVM training using SMO algorithm - DONE!")

    def i1_heuristic(
        self,
        i_2,
        error_cache
    ) -> int:

        """
        This heuristic finds E_1 such that |E_1 - E_2| is maximized (guarantees bigger step size).
        In order to avoid i_1 to stuck in case of large E_1, an error with opposite sign is selected.

        Arguments:
            i_2 : Selected i_2

            error_cache: A vector of last calculated error per samples
        
        Return:
            i_1 : sample index
        """

        E_2 = error_cache[i_2]

        # Restrict to only non-bounded alphas
        non_bounded_idx = np.argwhere((0 < self.alpha) & (self.alpha < self.c)).reshape((1, -1))[0]

        if non_bounded_idx.shape[0] > 0:
            if E_2 >= 0:
                i_1 = non_bounded_idx[np.argmin(error_cache[non_bounded_idx])]
            else:
                i_1 = non_bounded_idx[np.argmax(error_cache[non_bounded_idx])]
        else:
            i_1 = np.argmax(np.abs(error_cache - E_2))

        return i_1

    def i2_heuristic(
        self,
        non_kkt_array: np.ndarray
    ) -> int:

        """
        This heuristic chooses the index of alpha_2 to optimize such that it violates the KKT conditions
        (up to kkt_thr).

        Arguments:
            non_kkt_array: indices of alphas that don't satisfy the KKT conditions,
            from which the heuristic selects

        Return:
            i_2:  index of selected alpha, if all alphas satisfies KKT reutrn -1

            non_kkt_array: updated non_kkt_array
        """

        # Initialize i_2 to -1 in case all alphas satisfy KKT conditions
        i_2 = -1

        # Select the first alpha that violates KKT conditions
        # Note: In this case, better use a for loop (only worst case will iterate ovel all samples)
        for idx in non_kkt_array:
            # First remove it from array (eliminate stucking on the same sample)
            non_kkt_array = np.delete(non_kkt_array, np.argwhere(non_kkt_array == idx))

            # Check KKT
            if not self.check_kkt(idx):
                i_2 = idx
                break

        if i_2 == -1:
            # All given samples satisfies KKT condition
            # Recheck KKT on all samples
            idx_array = np.arange(self.alpha.shape[0])
            non_kkt_array = idx_array[~(self.check_kkt(idx_array))]
            if non_kkt_array.shape[0] > 0:
                # There are still samples to optimize
                # Shuffle array for randomness
                np.random.shuffle(non_kkt_array)
                i_2 = non_kkt_array[0]
                non_kkt_array = non_kkt_array[1:-1]

        return i_2, non_kkt_array

    def check_kkt(
        self,
        check_idx: int
     ) -> np.ndarray:

        """
        This function checks if sample_idx satisfies KKT conditions.

        Arguments:
            check_idx: Indices of alphas to check (scalar or vector)

        Return:
            kkt_condition_satisfied: Boolean array per alpha
        """

        alpha_idx = self.alpha[check_idx]
        _, score_idx = self.predict(self.support_vectors[check_idx, :])
        y_idx = self.support_labels[check_idx]
        r_idx = y_idx * score_idx - 1
        cond_1 = (alpha_idx < self.c) & (r_idx < - self.kkt_thr)
        cond_2 = (alpha_idx > 0) & (r_idx > self.kkt_thr)

        return ~(cond_1 | cond_2)

    def compute_boundaries(
        self,
        alpha_1,
        alpha_2,
        y_1,
        y_2
    ) -> Tuple[float, float]:

        """"
        Computes the lower and upper bounds for alpha_2.

        Arguments:
            alpha_1, alpha_2: Values before optimization

            y_1, y_1: labels of x[i1,:] and x[i2,:]

        Return:
            lb: Lower bound of alpha_2

            ub: Upper bound of alpha_2
        """

        if y_1 == y_2:
            lb = np.max([0, alpha_1 + alpha_2 - self.c])
            ub = np.min([self.c, alpha_1 + alpha_2])
        else:
            lb = np.max([0, alpha_2 - alpha_1])
            ub = np.min([self.c, self.c + alpha_2 - alpha_1])
        return lb, ub

    def compute_eta(
        self,
        x_1,
        x_2
    ) -> float:

        """
        Computes eta = K(x_1,x_1) + K(x_2,x_2) - 2K(x_1,x_2)

        Arguments:
            x_1, x_2: feature vectors of samples i1, i2

        Return:
            eta
        """
        
        #return self.kernel(x_1, x_1) + self.kernel(x_2, x_2) - 2*self.kernel(x_1, x_2)
        return self.kernel.calc(x_1, x_1) + self.kernel.calc(x_2, x_2) - 2*self.kernel.calc(x_1, x_2)
    
    def compute_b(
        self,
        alpha_1_new,
        alpha_2_new,
        E_1,
        E_2,
        i_1,
        i_2
    ) -> None:

        """"
        Computes the updated threshold b.
        Uses the same method as in the original paper.

        Arguments:
            alpha_1_new: New value of alpha_1

            alpha_2_new: New value of alpha_2

            E_1: Difference between SVM's prediction and label

            E_2:  Difference between SVM's prediction and label

            i_1: Index of alpha_1

            i_2: Index of alpha_2
        """

        x_1 = self.support_vectors[i_1]
        x_2 = self.support_vectors[i_2]

        b1 = self.b - E_1 - self.support_labels[i_1] * (alpha_1_new - self.alpha[i_1]) * self.kernel.calc(x_1, x_1) - \
            self.support_labels[i_2] * (alpha_2_new - self.alpha[i_2]) * self.kernel.calc(x_1, x_2)

        b2 = self.b - E_2 - self.support_labels[i_1] * (alpha_1_new - self.alpha[i_1]) * self.kernel.calc(x_1, x_2) - \
            self.support_labels[i_2] * (alpha_2_new - self.alpha[i_2]) * self.kernel.calc(x_2, x_2)

        if 0 < alpha_1_new < self.c:
            self.b = b1
        elif 0 < alpha_2_new < self.c:
            self.b = b2
        else:
            self.b = np.mean([b1, b2])

    def rbf_kernel(self, u, v):

        """
        RBF kernel implementation, i.e. K(u,v) = exp(-gamma_rbf*|u-v|^2).
        gamma_rbf is a hyper parameter of the model.

        Arguments:
            u: an (N,) vector or (N,D) matrix,

            v: if u is a vector, v must have the same dimension
               if u is a matrix, v can be either an (N,) or (N,D) matrix.

        Return:
            K(u,v): kernel matrix as follows:
                    case 1: u, v are both vectors:
                        K(u,v): a scalar K=u.T*v

                    case 2: u is a matrix, v is a vector
                        K(u,v): (N,) vector, the i-th element corresponds to K(u[i,:],v)

                    case 3: u and V are both matrices
                        k(u,v): (N,D) matrix, the i,j entry corresponds to K(u[i,:],v[j,:])
        """

        # In case u, v are vectors, convert to row vector
        if np.ndim(v) == 1:
            v = v[np.newaxis, :]

        if np.ndim(u) == 1:
            u = u[np.newaxis, :]

        # Broadcast to (N,D,M) array
        # Element [i,:,j] is the difference between i-th row in u and j-th row in v
        # Squared norm along second axis, to get the norm^2 of all possible differences, results an (N,M) array
        dist_squared = np.linalg.norm(u[:, :, np.newaxis] - v.T[np.newaxis, :, :], axis=1) ** 2
        dist_squared = np.squeeze(dist_squared)

        return np.exp(-self.gamma_rbf * dist_squared)

    @staticmethod
    def linear_kernel(
        u,
        v
    ) -> np.ndarray:

        """
        Linear kernel implementation.

        Arguments:
            u: an (N,) vector or (N,D) matrix,

            v: if u is a vector, v must have the same dimension
               if u is a matrix, v can be either an (N,) or (N,D) matrix.

        Return:
            K(u,v): kernel matrix as follows:
                    case 1: u, v are both vectors:
                        K(u,v): a scalar K=u.T*v

                    case 2: u is a matrix, v is a vector
                        K(u,v): (N,) vector, the i-th element corresponds to K(u[i,:],v)

                    case 3: u and V are both matrices
                        k(u,v): (N,D) matrix, the i,j entry corresponds to K(u[i,:],v[j,:])
        """

        return np.dot(u, v.T)


class OneVsAllClassifier:

    """
    Implements a one-vs-all strategy for multi-class classification.
    """

    def __init__(
        self,
        solver: Solver,
        num_classes: int,
        **kwargs
    ) -> None:

        """
        Arguments:
            solver : solver class, e.g., SVM

            num_classes : numer of classes

            kwargs : keyword arguments passed to each solver instance
            
        """

        self._binary_classifiers = [solver(**kwargs) for i in range(num_classes)]
        self._num_classes = num_classes

    def predict(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        """
        Arguments:
            x : (N,D) data matrix, each row is a sample.

        Return:
            pred : predictions, the i-th element corresponds to the i-th sample, y={-1,+1}
        """
        
        n = x.shape[0]
        scores = np.zeros((n, self._num_classes))

        for idx in range(self._num_classes):
            scores[:,idx] = self._binary_classifiers[idx].predict(x)[1]

        pred = np.argmax(scores, axis=1)

        return pred

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:

        """
        Train One-vs-All classifier.

        Arguments:
            x_train : (N,D) data matrix, each row is a sample.

            y_train : Labels vector, y must be {-1,1}
        """

        for idx in range(self._num_classes):
            # Convert labels to binary {+1,-1}
            y_tmp = 1.*(y_train == idx) - 1.*(y_train != idx)

            # Train a binary classifier
            self._binary_classifiers[idx].fit(x_train, y_tmp)