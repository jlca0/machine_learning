import numpy as np

class QRDecomposition:
    def __init__(
        self, 
        A: np.ndarray
    ) -> None:
        """
        Initializes a class that encodes the QR decomposition of a matrix of real numbers
        making use of Householder reflexions.

        Atributes:
            H: (np.ndarray) A matrix with its upper half containing the R matrix and its
                                     lower half containing each step's reflexion column vector.
            R_diag: (np.ndarray) The R matrix diagonal, overwritten by the reflexion vectors.
                                
                                    
        Parameters:
            A: (np.ndarray) The matrix of real numbers with decompostion QR.     
        """
        self.H = A.copy()
        n, m = self.H.shape
        columns = min(n,m)
        self.R_diag = np.zeros(columns)
        self.scale = np.zeros(columns)
        
        for k in range(columns):
            scale, r_diag = self._get_householder_reflexion(k)
            self._apply_householder_reflexion(self.H, k, scale)
            self.R_diag[k] = r_diag
            self.scale[k] = scale

    def get_R(self) -> np.ndarray:
        """
        Creates an array for the R matrix of the QR decomposition from the QRDecomposition.H matrix.

        Returns:
            R: (np.ndarray) R matrix of the QR decomposition 
        """
        R = self.H.copy()
        n = len(self.R_diag)
        for k in range(n):
            R[k, k] = self.R_diag[k]
        return np.triu(R) 

    def get_Q(self) -> np.ndarray:
        """
        Creates an array for the Q matrix of the QR decomposition from the QRDecomposition.H matrix.

        Returns:
            Q: (np.ndarray) Q matrix of the QR decomposition 
        """
        m, n = self.H.shape
        Q = np.eye(m)

        for k in range(n-1, -1, -1):
            v = self.H[k:, k].reshape(-1, 1)
            scale = self.scale[k]
            Q[k:, :] -= scale * (v @ (v.T @ Q[k:, :]))       
        return Q
    
    def get_condition_number(
        self, 
        max_iter: int = 5
    ) -> float:
        """
        Returns the condition number of the matrix A, using the Higham estimator for
        the 1-norm of the inverse.

        Parameters:
            max_iter: (Optional[int]) Maximum number of iterations for the Higham method.
    
        Returns:
            cond: (float) Condition number of A.
        """
        R = self.get_R()
        n = R.shape[1]
        R_square = R[:n, :n]  # ← solo la parte cuadrada superior
        
        norm_A = np.max(np.sum(np.abs(R_square.T @ R_square), axis=0))
        norm_inv_A = self._Higham_estimator(R_square, max_iter)
        
        return norm_A * norm_inv_A

    def solve(
        self,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Given a matrix of column vectors, it solves the linear system for the design matrix
        defined by A using its QR decomposition.

        Parameters:
            B: (np.ndarray) Matrix of independent terms by columns.

        Returns:
            X: (np.ndarray) Matrix of solutions by columns.
        """
        R = self.get_R()
        n = R.shape[1]
        R_square = R[:n, :n]  # ← solo la parte cuadrada
        X = self._downwards_solve(R_square.T, B)
        X = self._upwards_solve(R_square, X)
        return X

    def apply_Qt(
        self, 
        B: np.ndarray
    ) -> np.ndarray:
        """Applies Q.T to a matrix B without computing Q explicitly
    
        Parameters:
            B: (np.ndarray) Matrix of independent terms by columns.

        Returns:
            QtB: (np.ndarray) Modified matrix of independent terms by columns.
        """
        QtB = B.copy()
        for k in range(min(self.H.shape)):
            v = self.H[k:, k].reshape(-1, 1)
            scale = self.scale[k]
            QtB[k:, :] -= scale * (v @ (v.T @ QtB[k:, :]))
        return QtB
    
    def _get_householder_reflexion(
            self, 
            k:int
        )->(float,float):
        """
        Given a column k it returns the parameter scale of the Housholder reflexion matrix for that step
        and the single entry of the projection of the kth column. It modifies QRDecomposition.H setting the
        kth column lower half to the Housholder reflexion vector for that step.

        Parameters:
            k: (int) The number of the column.

        Returns:
            scale: (float) The sacle paraeter for the reflexion.
            R_diag: (float) Single entry of the projection of the kth column
        """
        v = self.H[k:, k]
        norm = np.linalg.norm(v)

        R_diag = - np.sign(v[0]) * norm
        v[0] -= R_diag
        scale = 2 / np.dot(v, v)
        
        return scale, R_diag

    def _apply_householder_reflexion(
            self, 
            H:np.ndarray[np.ndarray[float]], 
            k:int, 
            scale:float
        )->None:
        """
        It applies a given Householder reflexion scale parameter to the kth submatrix of H, modifying it.

        Parameters:
            H: (np.ndarray) The matrix being manipulated.
            k: (int) The kth submatrix index.
            scale: (float) Householder reflexion scale parameter.
        """
        v = H[k:, k]
        submatrix = H[k:, k+1:]
        if submatrix.size > 0:  # Check the submatrix dimensions, for the tall matrix case.
            submatrix -= scale * np.outer(v, np.dot(v, submatrix))

    def _downwards_solve(
        self,
        R: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Given an lower triangular matrix and a matrix of independent terms, it solves
        the system downwards.

        Parameters: 
            R: (np.ndarray) Lower triangular matrix. 
            B: (np.ndarray) Matrix of independent terms by columns.

        Returns:
            X: (np.ndarray) Matrix of solutions by columns.
        """
        m, n = R.shape
        p, q = B.shape
        X = np.zeros((n,q))
        for k in range(n):
            X[k, :] = B[k, :]
            if k != 0:
                X[k, :] -= R[k, :k]@X[:k, :]
            X[k, :] = X[k, :]/R[k, k]
        return X

    def _upwards_solve(
        self,
        R: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Given a upper triangular matrix and a matrix of independent terms, it solves
        the system upwards.

        Parameters: 
            R: (np.ndarray) Upper triangular matrix. 
            B: (np.ndarray) Matrix of independent terms by columns.

        Returns:
            X: (np.ndarray) Matrix of solutions by columns.
        """
        m, n = R.shape
        p, q = B.shape
        X = np.zeros((n,q))
        for k in range(n-1,-1,-1):
            X[k, :] = B[k, :]
            if k != n-1:
                X[k, :] -= R[k, k+1:]@X[k+1:, :]
            X[k, :] = X[k, :]/R[k, k]
        return X
    
    def _Higham_estimator(
        self, 
        R: np.ndarray, 
        max_iter: int = 5
    ) -> float:
        """
        Given an upper triangular matrix it computes the 1-norm of its inverse
        by Higham's algorithm.

        Parameters:
            R: (np.ndarray) Upper triangular matrix.
            max_iter: (Optional[int]) Maximum number of iterations for the Higham method.

        Returns:
            norm: 1-norm of R inverse
        """
        n = R.shape[1]
        k = 0
        initial_vector = np.full((n, 1), 1/n)
        vector = self._upwards_solve(R, self._downwards_solve(R.T, initial_vector))
        

        while k < max_iter and np.max(np.abs(vector)) > (vector.T @ initial_vector):
            direction = np.sign(vector)
            proof_vector = self._upwards_solve(R, self._downwards_solve(R.T, direction))

            j = np.argmax(np.abs(proof_vector))
            initial_vector = np.zeros((n, 1))
            initial_vector[j] = 1.0
            
            vector = self._upwards_solve(R, self._downwards_solve(R.T, initial_vector))
            k += 1
            
        return np.sum(np.abs(vector))

class SVD():
    def __init__(self, A: np.ndarray):
        """
        Initializes a SVD object which enables to compute the Singular Value Decomposition (SVD) of a given matrix.

        Attributes:
            A: (np.ndarray) The matrix to be decomposed.
        """
        self.A = A
        self.nrows, self.ncol = np.shape(self.A)

    def decompose(self):
        """Performs the SVD of the matrix making use of the Jacobi algorithm and a classic pivot strategy.
        
        Returns:
            U: (np.ndarray) Left eigenvector matrix by columns.
            L: (np.ndarray) Vector of singular values (not sorted).
            W: (np.ndarray) Right eigenvector matrix by columns.
        """
        varA = self.A.T @ self.A
        sing_values, V = self._jacobi(varA, tol = 1e-10, iter_max = 100 * self.nrows**2)
        L = np.sqrt(sing_values)
        U = self.A @ V @ np.diag(1/L)
        W = V
        self._show_results([L,U,W], 'Singular')
        return U, L, W

    def _jacobi(self, matrix:np.ndarray, tol: float, iter_max: int, internal_call = True):
        """Computes the spectral decomposition of a symmetric matrix making use of the Jacobi algorithm with 
        classic pivot strategy.
        
        Input:
            matrix: (np.ndarray) The matrix to be decomposed.
            tol: (float) Error tolerance of the iterative method.
            iter_max: (int) Maximum number of iterations.
            internal_call: (bool) Flag for turning off verbose mode.

        Returns:
            np.diag(A0): (np.ndarray) Vector of eigenvalues.
            R: (np.ndarray) Matrix of eigenvectors by columns.
        """
        n, m = np.shape(matrix)
        A0 = matrix
        O = np.eye(n)
        R = O
        error = tol + 1
        cont = 0
        while error >= tol and cont < iter_max:
            p, q = self._search_pivot(A0)
            error = abs(A0[p, q])    
            O = self._rotation_matrix(p, q, A0)
            # main iteration
            A0 = O.T @ A0 @ O
            R = R @ O
            cont += 1
        if not internal_call:
            self._show_results([A0, R],'Eigen')
        return np.diag(A0), R
    
    def _search_pivot(self, A0: np.ndarray):
        """Given a matrix it picks the greatest non-diagonal entry by absolute value.
        
        Input:
            A0: (np.ndarray) The matrix being searched.

        Returns:
            p: (int) The pivot is in the pth row.
            q: (int) The pivot is in the qth column.  
        """
        n, m = np.shape(A0)
        pivot = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                if abs(A0[i, j]) > pivot:
                    pivot = abs(A0[i, j])
                    p, q = i, j
        return p, q
    
    def _rotation_matrix(self, p: int, q: int, A0: np.ndarray):
        """
        It computes the Givens rotation matrix of a symmetric matrix given a selected entry.

        Input:
            p: (int) The first coordinate of the selected entry.
            q: (int) The second coordinate of the selected entry.
            A0: (np.ndarray) A symmetric matrix.

        Returns:
            O: (np.ndarray) Givens rotation matrix.
        """
        n, m = np.shape(A0)
        if A0[q,q]-A0[p,p] > 1e-16:
            theta = 0.5*np.arctan(2*A0[p,q]/(A0[q,q]-A0[p,p]))
        else:
            theta = np.pi/4
        O = np.eye(n)
        O[p,q] = np.sin(theta)
        O[q,p] = -np.sin(theta)
        O[q,q] = np.cos(theta)
        O[p,p] = np.cos(theta)
        return O

    def _show_results(self, display: list, case: str):
        """
        Once a decomposition is performed it shows its result on screen.

        Input:
            display: (list) A list of matrices/vectors.
            case: (str) A flag indicating whether a spectral or 
                        singular value decomposition is being displayed.
        """
        if case == 'Eigen':
            print(f'Eigenvalues: {np.diag(display[0])} \n')
            print(f'Eigenvectors: {display[1]} \n')
        elif case == 'Singular':
            print(f'Singular values: {display[0]} \n')
            print(f'Left eigenvectors: {display[1]} \n')
            print(f'Right eigenvectors: {display[2]} \n')

if __name__ == "__main__": 
    A = np.array([[2, 1, 0, 4],
                [1, 1, 3, 7],
                [0, 3, 7, 2],
                [4, 7, 2, 5]], dtype=float)

    svd = SVD(A)
    print('This is the SVD decomposition of the matrix A: \n')
    print(f'{A} \n')
    svd.decompose()
    