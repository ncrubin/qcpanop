import numpy as np
from qcpanop.algebra.majorana import omega_transform

def test_omega_transform():
    dim = 4
    omega = omega_transform(dim)
    assert np.allclose(omega.conj().T @ omega, np.eye(2 * dim))

def quad_H():
    dim = 2
    A = np.random.randn(dim**2).reshape((dim, dim)) + 1j * np.random.randn(dim**2).reshape((dim, dim))
    A = A + A.conj().T
    B = np.random.randn(dim**2).reshape((dim, dim))  #+ 1j * np.random.randn(dim**2).reshape((dim, dim))
    B = B - B.conj().T
    np.fill_diagonal(B, 0)
    assert np.allclose(B.T, -B)

if __name__ == "__main__":
    test_omega_transform()
    quad_H()