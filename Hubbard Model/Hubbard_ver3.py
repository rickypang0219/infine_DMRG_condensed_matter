import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import  kron, eye
from scipy.sparse.linalg import eigsh
from numpy import linalg

# Parameters
LatticeLength = 8
MaximalStates = 64

# Local dimension for a site
Dim = 4

# Define the interaction Strength
t = 2
U =  1
mu = 0

# To store the ground state energy of eahc DMRG step
E_GS = []
SE = []  # von Neumann entropy


# spin up
a_up = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

# spin down
a_down = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

# F = (1 - 2n_up) (1- 2n_down)
F = np.diag([1, -1, -1, 1])

# System elements
sysBlock_a_up = a_up
sysBlock_a_down = a_down
sysBlock_F = F
sysBlock_Length = 1

sysBlock_Ham = U * a_up.conj().T @ a_up @ a_down.conj().T @ a_down \
               - mu * ( a_up.conj().T @ a_up + a_down.conj().T @ a_down)

sysBlock_N = a_up.conj().T @ a_up + a_down.conj().T @ a_down

# Environment elements
envBlock_a_up = a_up
envBlock_a_down = a_down
envBlock_F = F
envBlock_Length = 1

envBlock_Ham = U * a_up.conj().T @ a_up @ a_down.conj().T @ a_down \
               - mu * ( a_up.conj().T @ a_up + a_down.conj().T @ a_down)

envBlock_N = a_up.conj().T @ a_up + a_down.conj().T @ a_down







### RDM function
def partial_trace(psi, n1, n2):
    # define the density matrix for ground state psi
    rho = psi @ psi.conj().T
    rho_tensor = rho.reshape(int(n1), int(n2), int(n1), int(n2))
    RDM_sys = np.trace(rho_tensor, axis1=1, axis2=3)
    RDM_env = np.trace(rho_tensor, axis1=0, axis2=2)
    return RDM_sys, RDM_env


### iDMRG Algorithm

while (sysBlock_Length + envBlock_Length) < LatticeLength:
    """ Construct a sysBlock_Ham which describes the sysBlock """
    # 1. Initialzed the dimension of sysBlock = kron(sysBlock_Ham, eye(Dim))
    # 2. Include the interaction of the site inside the sysBlock and the newly added site
    # 3. Describe their interactions as kron(sysBlock..., S...), we can understand it as sysBlock \otimes S(new site)

    """ Number operator , the meaning of sysBlock operators have not been changed in this step 
    If we put the Number op after the sysBlock_Ham, then it will gg since sysBlock_Ham dimension changes from 4**(n) to 4**(n+1) 

    """
    sysBlock_N = kron(sysBlock_N, eye(Dim)) + kron(eye(sysBlock_Ham.shape[0]), a_up.conj().T @ a_up + a_down.conj().T @ a_down)

    # consider the sysBlock already have two sites
    """ Don't know why the on-site spin spin interaction term is not hermition 
    To force the Hamiltonian becomes Hermitian, we H = 0.5( H + H.conj().T) 

    Seems even in t is non-zero, the sysBlock_Ham is not hermitian already. But interestingly the envBlock is hermitian......
    mu is ok, but t and U terms fuck up...... 
    In the 1st loop: Total Length = 6, such weird thing happens 
    when t != 0,  sysBlock_ham is not hermitian,  envBlock_ham is hermitian
    when U != 0 , sysBlock_ham is hermitian , envBlock_ham is not hermitian  
    """
    sysBlock_Ham = kron(sysBlock_Ham, eye(Dim)) \
                   - t * (kron(sysBlock_a_up.conj().T @ sysBlock_F, a_up) + kron(sysBlock_a_down.conj().T, F @ a_down)) \
                   - t * (-1 * kron(sysBlock_a_up @ sysBlock_F, a_up.conj().T) - 1 * kron(sysBlock_a_down, F @ a_down.conj().T)) \
                   + U * kron(eye(sysBlock_Ham.shape[0]), a_up.conj().T @ a_up @ a_down.conj().T @ a_down) \
                   - mu * kron(eye(sysBlock_Ham.shape[0]), a_up.conj().T @ a_up + a_down.conj().T @ a_down)

    # Update the envBlock: add a site on LHS ( this update part could be done in a clever way using kron?)

    envBlock_N = kron(eye(Dim), envBlock_N) + kron(a_up.conj().T @ a_up + a_down.conj().T @ a_down, eye(envBlock_Ham.shape[0]))

    # eye(envBlock_Ham.shape[0]) means the identity with  envBlock_Ham size
    envBlock_Ham = kron(eye(Dim), envBlock_Ham) \
                   - t * (kron(a_up.conj().T @ F, envBlock_a_up) + kron(a_down.conj().T, envBlock_F @ envBlock_a_down)) \
                   - t * (-1 * kron(a_up @ F, envBlock_a_up.conj().T) - 1 * kron(a_down, envBlock_F @ envBlock_a_down.conj().T)) \
                   + U * kron(a_up.conj().T @ a_up @ a_down.conj().T @ a_down, eye(envBlock_Ham.shape[0])) \
                   - mu * kron(a_up.conj().T @ a_up + a_down.conj().T @ a_down, eye(envBlock_Ham.shape[0]))



    # Make sure the Hamiltonian that we defined are Hermitian:
    sysBlock_Ham = (sysBlock_Ham + sysBlock_Ham.conj().T) / 2
    envBlock_Ham = (envBlock_Ham + envBlock_Ham.conj().T) / 2

    # Update the length
    sysBlock_Length = sysBlock_Length + 1
    envBlock_Length = envBlock_Length + 1

    """Construct the spin oerators of the newly added sites on middle: """
    # sysBlock_Splus = I \otimes Splus ; envBlock_Splus = Splus \otimes I s.t.
    # Interaction of these two point = I \otimes Splus \otimes Splus \otimes I

    sysBlock_a_up = kron(eye(len(sysBlock_Ham.toarray()) / Dim), a_up)
    sysBlock_a_down = kron(eye(len(sysBlock_Ham.toarray()) / Dim), a_down)
    sysBlock_F = kron(eye(len(sysBlock_Ham.toarray()) / Dim), F)

    envBlock_a_up = kron(a_up, eye(len(envBlock_Ham.toarray()) / Dim))
    envBlock_a_down = kron(a_down, eye(len(envBlock_Ham.toarray()) / Dim))
    envBlock_F = kron(F, eye(len(envBlock_Ham.toarray()) / Dim))


    print("Current Length", sysBlock_Length + envBlock_Length )
    """ Merger two blocks together to form a superblock """
    # 1. New sysBlock = sysBlock \otimes I_{dim = sysBlock} and similar for envBlock
    # 2. Include the interaction of the middle sites: Use Heisenberg Hamiltonian:
    # 3. Add all stuff together to form the superblock Hamiltonian

    # In the black box picture, we can replace
    Hsuper = kron(envBlock_Ham, eye(len(envBlock_Ham.toarray()))) + kron(eye(len(sysBlock_Ham.toarray())), envBlock_Ham) \
             - t * (kron(sysBlock_a_up.conj().T @ sysBlock_F, envBlock_a_up) + kron(sysBlock_a_down.conj().T,envBlock_F @ envBlock_a_down)) \
             - t * (-1 * kron(sysBlock_a_up @ sysBlock_F, envBlock_a_up.conj().T) - 1 * kron(sysBlock_a_down,envBlock_F @ envBlock_a_down.conj().T))


    Hsuper = 0.5 * ( Hsuper + Hsuper.conj().T )

    print("Required Ram (in Gb) = " + str(Hsuper.toarray().nbytes/ 10**9 ) )


    # Define Number Operator for checking the filling:
    Nsuper = kron(sysBlock_N, eye(len(envBlock_Ham.toarray()))) + kron(eye(len(sysBlock_Ham.toarray())), envBlock_N)

    """ Return the ground state of superblock """
    # the vec_GS is (n, 1) array, not a (1,n) array as usual
    # If we need to extract the whole spectrum, we need to find all eigenvales but not the smallest one
    val_GS, vec_GS = eigsh(Hsuper, 1, which="SA", return_eigenvectors=True)

    """ Check Error
    Error = < psi | H^{2} | psi > - ( < psi | H | psi > )^{2} 
    """
    error = vec_GS.conj().T @ (Hsuper @ Hsuper) @ vec_GS - (vec_GS.conj().T @ Hsuper @ vec_GS) ** 2
    print("Error =", error)

    # Store the ground state energy of superBlock under each iteration
    E_GS.append(val_GS / (sysBlock_Length + envBlock_Length))

    # Check the convergence of GS energy
    print("Ground State Energy = " + str(val_GS / (sysBlock_Length + envBlock_Length)))

    # GS expectation value of number operator
    print(r"GS expectation value of Number operator = " + str ((vec_GS.conj().T @ Nsuper @ vec_GS)[0]))

    """Compute the parital trace of the ground state density matrix. These can be computed by tensor network """

    sysBlock_DM, envBlock_DM = partial_trace(vec_GS, len(sysBlock_Ham.toarray()), len(envBlock_Ham.toarray()))
    print("Check Trace", np.trace(sysBlock_DM), np.trace(envBlock_DM))

    # Ensure the RDM of each Block is Hermitian
    sysBlock_DM = (sysBlock_DM + sysBlock_DM.conj().T) / 2
    envBlock_DM = (envBlock_DM + envBlock_DM.conj().T) / 2

    """ Diagonalize the reduced density matrix """
    sysBlock_weight, sysBlock_rotationMatrix = np.linalg.eig(sysBlock_DM)
    envBlock_weight, envBlock_rotationMatrix = np.linalg.eig(envBlock_DM)

    """ Sorted Method using Numpy 
    np.linalg.eig is suck in the sense that it does not sort the eigenvalues from largest to smallest algebraically. 
    We need to use argsort() to ensure the sorting 
    
    """


    # This part can be done with using argsort
    # Then idx tells us the descending order of weights of RDM
    sysBlock_idx = sysBlock_weight.argsort()[::-1]
    sysBlock_weight_sort = sysBlock_weight[sysBlock_idx]
    Isys = sysBlock_idx  # call this idex and Isys


    # Check whether the RDM will equal [1,0,0,0,.....], implying the target state is not entangled state
    print("sysBlock Weights=", np.real(sysBlock_weight_sort[:min(len(sysBlock_weight_sort), 8)]))



    # von Neumann entropy of sysBlock
    S = -(sysBlock_weight_sort * np.log(sysBlock_weight_sort)).sum()
    SE.append(S)

    # Similar for the environment:
    envBlock_idx = envBlock_weight.argsort()[::-1]
    envBlock_weight_sort = envBlock_weight[envBlock_idx]
    Ienv = envBlock_idx

    """ Truncation of the Hamiltonian"""
    # Obtain the new set of bases with highest weightings
    # If the dimension of eigenvector matrix is larger than the Maximum States, then we set a cut off here
    # Only extract the states with higher weightings
    # Isys[:min(MaximalStates, len(sysBlock_rotationMatrix) ) ] scans the higher weightings column vector of the eigenvector matrix

    sysBlock_rotationMatrix = sysBlock_rotationMatrix[:, Isys[:min(MaximalStates, len(sysBlock_rotationMatrix))]]
    envBlock_rotationMatrix = envBlock_rotationMatrix[:, Ienv[:min(MaximalStates, len(envBlock_rotationMatrix))]]

    # Project the Hamiltonian into lower dimensional subspace spanned by sysBlock_rot_matrix
    # sysBlock:

    sysBlock_Ham = sysBlock_rotationMatrix.conj().T @ sysBlock_Ham.toarray() @ sysBlock_rotationMatrix
    sysBlock_a_up = sysBlock_rotationMatrix.conj().T @ sysBlock_a_up.toarray() @ sysBlock_rotationMatrix
    sysBlock_a_down = sysBlock_rotationMatrix.conj().T @ sysBlock_a_down.toarray() @ sysBlock_rotationMatrix
    sysBlock_F = sysBlock_rotationMatrix.conj().T @ sysBlock_F.toarray() @ sysBlock_rotationMatrix
    sysBlock_N = sysBlock_rotationMatrix.conj().T @ sysBlock_N.toarray() @ sysBlock_rotationMatrix

    envBlock_Ham = envBlock_rotationMatrix.conj().T @ envBlock_Ham.toarray() @ envBlock_rotationMatrix
    envBlock_a_up = envBlock_rotationMatrix.conj().T @ envBlock_a_up.toarray() @ envBlock_rotationMatrix
    envBlock_a_down = envBlock_rotationMatrix.conj().T @ envBlock_a_down.toarray() @ envBlock_rotationMatrix
    envBlock_F = envBlock_rotationMatrix.conj().T @ envBlock_F.toarray() @ envBlock_rotationMatrix
    envBlock_N = envBlock_rotationMatrix.conj().T @ envBlock_N.toarray() @ envBlock_rotationMatrix


    # envBlock_a_up = sysBlock_a_up
    # envBlock_a_down = sysBlock_a_down
    # envBlock_F = sysBlock_F
    # envBlock_Ham = sysBlock_Ham
    # envBlock_N = sysBlock_N


    # sysBlock_a_up = envBlock_a_up
    # sysBlock_a_down = envBlock_a_down
    # sysBlock_F = envBlock_F
    # sysBlock_Ham = envBlock_Ham
    # sysBlock_N = envBlock_N








""" Plot Graph """
loop_array = np.arange(0, len(E_GS), 1)
length_array = np.arange(2, 2 * sysBlock_Length, 2)

fig, ax = plt.subplots()
plt.subplot(1, 2, 1)
plt.title(r"Convergence of Ground State Energy ")
plt.scatter(loop_array, np.array(E_GS), facecolors='none', edgecolors='b', label=r"DMRG GS")
plt.ylabel(r" Energy per Site")
plt.xlabel(r"Iterations")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Entanglement Entropy VS Number of Sites")
# plt.scatter(length_array, np.array(SE),facecolors='none', edgecolors='g')
plt.plot(length_array, np.array(SE), "-o", fillstyle='none', label=r"von Neumann Entropy")
plt.ylabel("Entropy")
plt.xlabel("Number of sites")
plt.legend()
np.save("SE_myDMRG", np.array(SE))
plt.show()

