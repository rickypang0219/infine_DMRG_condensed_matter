import matplotlib.pyplot as plt 
import numpy as np 
from scipy.sparse import kron , eye , identity
from scipy.sparse.linalg import eigsh 
from tqdm.auto import tqdm 


# Define the Operators inside the Heisenberg Model: 

# Spin Operator (set hbar = 1 ) 
Sx = 0.5 * np.array([[0,1],[1,0]])
Sy = 0.5 * np.array([[0, -1j],[1j, 0]])
Sz = 0.5 * np.array([[1,0],[0,-1]])


# Construct the S^{+} operator; S^{-} is the hermitian conjugate of S^{+}, therefore no need to define 
Splus = Sx + 1j * Sy

# Construct the Block and Environment of the System: 
# Suppose the dimension of the system and environment are 1 ( I have 1 sites initially in both Block and system)


# Block elements
sysBlock_Ham = np.zeros(Sx.shape)
sysBlock_Splus = Splus 
sysBlock_Sz = Sz 
sysBlock_Length = 1


# Environment elements: 
envBlock_Ham = np.zeros(Sx.shape)
envBlock_Splus = Splus 
envBlock_Sz = Sz 
envBlock_Length = 1


# Define the maximum number of Sites that we want to reach and the maximum number of lowest eigenstates that we would keep: 
LatticeLength = 16
MaximalStates = 64

# Local dimension for a site. In Heisenberg model, each site is described by a 2x2 matrix, impliying there are 2 states at one point 
Dim = 2

J = 1

# To store the ground state energy of eahc DMRG step
E_GS = [] 
SE = []  # von Neumann entropy 
 

# Tdqm Progress Bar module
times = 0
progress = tqdm(total=LatticeLength)

# Define a Partial Trace function
# This trace method has some bug
# def ptrace(psi, dim1, dim2):
#     # Perform the partil trace and obtain two reduced density matrix
#     rho = psi.reshape((dim2,dim1))
#     # @ = inner product key
#     # ndarray does not support transpose conjugate H, we need to use .conj.T explicitly
#     DM2 = rho @ rho.conj().T
#     DM1 = rho.conj().T @ rho
#     return DM1, DM2


def partial_trace(psi,n1,n2):
    # define the density matrix for ground state psi
    rho =  psi @ psi.conj().T
    rho_tensor = rho.reshape(int(n1), int(n2),int(n1),int(n2))
    RDM_sys = np.trace(rho_tensor, axis1=1, axis2=3)
    RDM_env = np.trace(rho_tensor, axis1=0, axis2=2)
    print("rho_shape= " + str(rho.shape) + "psi_shape = " + str(psi.shape))
    print("rho_reshaped_tensor = ", rho_tensor.shape)
    print("RDM shape=" , RDM_env.shape)
    return RDM_sys, RDM_env


# Next, follow the algorith of DMRG, we need to add a site on the left of the system and add a site on the right of environment: 

# Stop the while loop if (sys + env) length reach to 200
while (sysBlock_Length + envBlock_Length) <= LatticeLength:
    # Update the sysBlock: add a site on RHS 
    # kron(sysBlock_Ham, identity(Dim)) = initialize the sysBlock_Ham eg: 2 Dim case: 
    # kron(sysBlock_Ham, identity(Dim)) = 0_{4x4} matrix > initialize the sysBlock_Ham in 4D
    # print(kron(sysBlock_Splus, Splus.conj().T).toarray().shape)

    sysBlock_Ham = kron(sysBlock_Ham, eye(Dim))+\
                   J/2 * (kron(sysBlock_Splus, Splus.conj().T) + kron(sysBlock_Splus.conj().T, Splus)) + J * kron(sysBlock_Sz, Sz)

    #Update the envBlock: add a site on LHS ( this update part could be done in a clever way using kron?)
    envBlock_Ham = kron(eye(Dim), envBlock_Ham) + \
                   J/2 * (kron(Splus.conj().T,envBlock_Splus) + kron(Splus, envBlock_Splus.conj().T) ) + J * kron(Sz, envBlock_Sz)


    # Make sure the Hamiltonian that we defined are Hermitian:
    # print(sysBlock_Ham - sysBlock_Ham.conj().T)
    # print(envBlock_Ham - envBlock_Ham.conj().T)

    sysBlock_Ham = (sysBlock_Ham + sysBlock_Ham.conj().T) / 2
    envBlock_Ham = (envBlock_Ham + envBlock_Ham.conj().T) / 2


    # Update the operators of both sysBlock and envBlock: 
    # 1. Splus is a 2x2 matrix, then we add new site on the right > new Splus of system = Splus_{2x2} \otimes identity(len(sysBlock_Ham.toarray())/Dim)
    # at this time, sysBlock_Ham is a 4x4 matrix > New Splus = 4x4 matrix 

    # Block_Splus = kron(Splus, eye(len(sysBlock_Ham.toarray())/Dim))
    # sysBlock_Sz = kron(Sz, eye(len(sysBlock_Ham.toarray())/Dim))
    # sysBlock_Length = sysBlock_Length + 1
    #
    # # # Update the envBlock
    # envBlock_Splus = kron(eye(len(envBlock_Ham.toarray())/Dim), Splus)
    # envBlock_Sz = kron(eye(len(envBlock_Ham.toarray())/Dim), Sz)
    # envBlock_Length = envBlock_Length + 1
    
    # Zhihu Original code: Why not use is that I think the kron is wrong in the sense of its postition 
    #for sysBlock, it should be kron(sysBlock, id) rather than kron(id, sysBlock)
    sysBlock_Splus = kron(eye(len(sysBlock_Ham.toarray())/Dim), Splus)
    sysBlock_Sz = kron(eye(len(sysBlock_Ham.toarray())/Dim), Sz)
    envBlock_Splus = kron(Splus, eye(len(envBlock_Ham.toarray())/Dim))
    envBlock_Sz = kron(Sz,eye(len(envBlock_Ham.toarray())/Dim))

    sysBlock_Length = sysBlock_Length + 1
    envBlock_Length = envBlock_Length + 1




    # Merger two blocks together to form a superblock 
    # Initialize the Super Hamiltonian 
    # 1. sysBlock \otimes identity(envBlock)
    Hsuper = kron(sysBlock_Ham, eye(len(envBlock_Ham.toarray()))) + kron(eye(len(envBlock_Ham.toarray())), envBlock_Ham) \
        + J/2 * (kron(sysBlock_Splus.conj().T, envBlock_Splus) + kron(sysBlock_Splus, envBlock_Splus.conj().T)) + J * kron(sysBlock_Sz, envBlock_Sz)


    # Check Ram usage
    print("Required Ram (in Gb) = " + str(Hsuper.toarray().nbytes / 10 ** 9))


    # Return the ground state of superblock
    val_GS, vec_GS = eigsh(Hsuper, 1, which="SA", return_eigenvectors=True)


    print("GS shape =", vec_GS.shape)


    # Store the ground state energy of superBlock under each iteration
    E_GS.append( val_GS/(sysBlock_Length + envBlock_Length))
    print("GS Energy=", val_GS/(sysBlock_Length + envBlock_Length))
    sysBlock_DM, envBlock_DM = partial_trace(vec_GS,  len(sysBlock_Ham.toarray()), len(envBlock_Ham.toarray()))
    # Check trace of density matrix always = 1
    print("Check Trace", np.trace(sysBlock_DM), np.trace(envBlock_DM))



    """ Make Sure the matrix is Hermitian"""
    sysBlock_DM = (sysBlock_DM + sysBlock_DM.conj().T)/2
    envBlock_DM = (envBlock_DM + envBlock_DM.conj().T)/2


    # Diagonalize the reduced density matrix
    sysBlock_weight, sysBlock_rotationMatrix = np.linalg.eigh(sysBlock_DM)
    envBlock_weight, envBlock_rotationMatrix = np.linalg.eigh(envBlock_DM)
    
    """Make the sysBlock weigth array in descending order"""
    # sysBlock_weight_des = sorted(sysBlock_weight)[::-1]
    # Construct the RDM of system Block 
    # Isys = np.diag(sysBlock_weight_des)
    # sysBlock_val, sysBlock_vec = np.linalg.eig(Isys)
    # sysBlock_rotationMatrix = sysBlock_vec


    """ Sorted Method using Numpy """
    sysBlock_idx = sysBlock_weight.argsort()[::-1]
    sysBlock_weight_sort = sysBlock_weight[sysBlock_idx]
    # sysBlock_rotationMatrix_sort = sysBlock_rotationMatrix[:, sysBlock_idx]
    Isys = sysBlock_idx
    # sysBlock_rotationMatrix = sysBlock_rotationMatrix_sort


    """ Check Entanglement 
    If the resulted array is [1,0,0,.....,0], implying our target state is unentangled
    If the resulted array is [0.7, 0.2, 0.02, ... 1e-8] , implying our targert state is an enatangled state 
    Set len of max sysBlock_weight = 8 to check 
    """

    print("sysBlock Weights=", np.real(sysBlock_weight_sort[:min(len(sysBlock_weight_sort), 8) ]))

    # von Neumann entropy of sysBlock 
    S = -(sysBlock_weight_sort * np.log(sysBlock_weight_sort)).sum()
    SE.append(S)



    # # Similarly for environment Block
    # envBlock_weight_des = sorted(envBlock_weight)[::-1]
    # # envBlock_weight_des = sorted(envBlock_weight)[::-1]
    # Ienv = np.diag(envBlock_weight_des)
    # envBlock_val, envBlock_vec = np.linalg.eig(Ienv)
    # # envBlock_rotationMatrix = envBlock_vec


    envBlock_idx = envBlock_weight.argsort()[::-1]
    envBlock_weight_sort = envBlock_weight[envBlock_idx]
    # envBlock_rotationMatrix_sort = envBlock_rotationMatrix[:, envBlock_idx]
    Ienv = envBlock_idx
    # envBlock_rotationMatrix = envBlock_rotationMatrix_sort


    #Obtain the truncated basis( There is some bugs in the truncation)
    # sysBlock is a matrix contains eigenvector, but not an array
    sysBlock_rotationMatrix = sysBlock_rotationMatrix[:, Isys[:min(MaximalStates, len(sysBlock_rotationMatrix) ) ]  ]
    envBlock_rotationMatrix = envBlock_rotationMatrix[:, Ienv[:min(MaximalStates, len(envBlock_rotationMatrix) ) ]  ]



    # print( sysBlock_Ham - sysBlock_rotationMatrix.conj().T @ sysBlock_Ham.toarray() @ sysBlock_rotationMatrix )


    #Rotate (truncate) to the new basis
    # sysblock:
    # kind of doing projection?? restrict the size of sysBlock_Ham to the size of sysBlock_rot_matrix
    # Since sysBlock/ envBlock are sparse matrix, we need to use toarray() to convert them as ndarray
    # print(sysBlock_rotationMatrix.conj().T.shape,sysBlock_Ham.toarray().shape, sysBlock_rotationMatrix.shape )
    sysBlock_Ham = sysBlock_rotationMatrix.conj().T @ sysBlock_Ham.toarray() @ sysBlock_rotationMatrix
    sysBlock_Splus = sysBlock_rotationMatrix.conj().T @ sysBlock_Splus.toarray() @ sysBlock_rotationMatrix
    sysBlock_Sz = sysBlock_rotationMatrix.conj().T @ sysBlock_Sz.toarray() @ sysBlock_rotationMatrix
    # envBlock
    envBlock_Ham = envBlock_rotationMatrix.conj().T @ envBlock_Ham.toarray() @ envBlock_rotationMatrix
    envBlock_Splus = envBlock_rotationMatrix.conj().T @ envBlock_Splus.toarray() @ envBlock_rotationMatrix
    envBlock_Sz = envBlock_rotationMatrix.conj().T @ envBlock_Sz.toarray() @ envBlock_rotationMatrix



    # tqdm Progress Bar Module > replace print Total length 
    progress.update(2) # Since we add two sites in middle > update +2 
    times = times + 2 # Similar to above argument 
    # print("Total length =", sysBlock_Length + envBlock_Length)


loop_array = np.arange(0,len(E_GS),1)
length_array = np.arange(2,2*sysBlock_Length,2)


# Save array 
# np.save("GS_bond_dim=" + str(MaximalStates) + "_Length=" + str(LatticeLength) , np.array(E_GS))


fig , ax = plt.subplots()
plt.subplot(1,2,1)
plt.title(r"Convergence of Ground State Energy ")
plt.scatter(loop_array, np.array(E_GS) ,facecolors='none', edgecolors='b' ,label=r"DMRG GS" )
plt.axhline(y=-np.log(2) + 0.25, color='r', linestyle='-', label=r"True GS")
plt.ylabel(r" Energy per Site")
plt.xlabel(r"Iterations")
plt.legend()


plt.subplot(1,2,2)
plt.title("Entanglement Entropy VS Number of Sites")
# plt.scatter(length_array, np.array(SE),facecolors='none', edgecolors='g')
plt.plot(length_array, np.array(SE), "-o", fillstyle='none', label=r"von Neumann Entropy")
plt.plot(length_array, 1/6 * np.log(  length_array  ) ) 
plt.ylabel("Entropy")
plt.xlabel("Number of sites")
plt.legend()
np.save("SE_myDMRG", np.array(SE))
plt.show()