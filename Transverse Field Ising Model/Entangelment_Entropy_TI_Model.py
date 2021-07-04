import matplotlib.pyplot as plt 
import numpy as np 
from scipy.sparse import  identity, kron , eye
from scipy.sparse.linalg import eigsh 
from tqdm import tqdm

sigma_x =  np.array([[0,1],[1,0]]) # interchange sigma_x and sigma_z
# sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z =  np.array([[1,0],[0,-1]])  # interchange sigma_x and sigma_z

LatticeLength = 40  # Final Lattice Length 
MaximalStates = 32    # Maximum States for truncation 

J = 1  # Interaction Strength for x direction 
h = 1  # Interaction Strength  for z direction 

# """ Initialize the sysBlock and envBlock"""
# sysBlock_Ham = -h * sigma_z # Initially we have on-site potential term
# sysBlock_Sigma_x = sigma_x
# sysBlock_Sigma_z = sigma_z
# sysBlock_Length = 1

# # Environment elements: 
# envBlock_Ham = -h * sigma_z # Initially we have on-site potential term
# envBlock_Sigma_x = sigma_x 
# envBlock_Sigma_z = sigma_z
# envBlock_Length = 1

Dim = 2 # Local site dimension 

E_GS = []  # To store the ground state energy of eahc DMRG step

SE  = [] # von Neumann Entropy for sysBlock

def partial_trace(psi,n1,n2):
    # define the density matrix for ground state psi
    rho =  psi @ psi.conj().T
    rho_tensor = rho.reshape(int(n1), int(n2),int(n1),int(n2))
    RDM_sys = np.trace(rho_tensor, axis1=1, axis2=3)
    RDM_env = np.trace(rho_tensor, axis1=0, axis2=2)
    return RDM_sys, RDM_env


# iDMRG Main Calculation
# vary the J
J_array = np.linspace(0.4, 1.5 ,30)


for i in range(len(J_array)):
    print("Number of iteration", i) 
    SE_local = 0 
    E_GS_local = 0
    """ Initialize the sysBlock and envBlock"""
    sysBlock_Ham = -h * sigma_z # Initially we have on-site potential term
    sysBlock_Sigma_x = sigma_x
    sysBlock_Sigma_z = sigma_z
    sysBlock_Length = 1

    # Environment elements: 
    envBlock_Ham = -h * sigma_z # Initially we have on-site potential term
    envBlock_Sigma_x = sigma_x
    envBlock_Sigma_z = sigma_z
    envBlock_Length = 1

    # while (sysBlock_Length + envBlock_Length) < LatticeLength:
    #loop time = length /2

    for _ in range(0,int(LatticeLength/2)):
        # Step 1: Enlarge the Blocks by adding a new site 

        """ To check whether the dimensions of operators are correct"""
        sysBlock_Ham = kron(sysBlock_Ham, eye(Dim))  -h * kron(eye(len(sysBlock_Ham)), sigma_z) \
            -J_array[i] * kron(sysBlock_Sigma_x, sigma_x)

        envBlock_Ham = kron(eye(Dim), envBlock_Ham)  -h * kron(sigma_z, eye(len(envBlock_Ham)))\
            -J_array[i] * kron(sigma_x, envBlock_Sigma_x)


        # Make sure both sysBlock and envBlock are Hermitian 
        sysBlock_Ham = 0.5 * ( sysBlock_Ham + sysBlock_Ham.conj().T )
        envBlock_Ham = 0.5 * ( envBlock_Ham + envBlock_Ham.conj().T )


        # Step 2: Perpare the operator for Superblock Hamiltonain 
        # The operators for middle two points 
        sysBlock_Sigma_x = kron(eye(len(sysBlock_Ham.toarray())/Dim), sigma_x)
        sysBlock_Sigma_z = kron(eye(len(sysBlock_Ham.toarray())/Dim), sigma_z)

        envBlock_Sigma_x = kron(sigma_x, eye(len(envBlock_Ham.toarray())/Dim))
        envBlock_Sigma_z = kron(sigma_z,eye(len(envBlock_Ham.toarray())/Dim))

        # Update the size of both blocks 
        sysBlock_Length = sysBlock_Length + 1
        envBlock_Length = envBlock_Length + 1

        # Step 3: Construct the Superblock Hamiltonain 
        # print(len(sysBlock_Ham.toarray()))
        H_super = kron(sysBlock_Ham, eye(len(envBlock_Ham.toarray()) )) + kron(eye(len(sysBlock_Ham.toarray())), envBlock_Ham) -J_array[i] * kron(sysBlock_Sigma_x,envBlock_Sigma_x)


        # Return the ground state of superblock
        val_GS, vec_GS = eigsh(H_super, 1, which="SA", return_eigenvectors=True)
        E_GS_local = val_GS/(sysBlock_Length + envBlock_Length)
        # print(val_GS/(sysBlock_Length + envBlock_Length))




        # Step 4: Construct the RDM for sysBlock and envBlock 
        sysBlock_DM, envBlock_DM = partial_trace(vec_GS,  len(sysBlock_Ham.toarray()), len(envBlock_Ham.toarray()))
        # Check trace of density matrix always = 1
        # print(np.trace(sysBlock_DM), np.trace(envBlock_DM))

        """ Make Sure the matrix is Hermitian"""
        sysBlock_DM = (sysBlock_DM + sysBlock_DM.conj().T)/2
        envBlock_DM = (envBlock_DM + envBlock_DM.conj().T)/2


        # Diagonalize the reduced density matrix
        sysBlock_weight, sysBlock_rotationMatrix = eigsh(sysBlock_DM)
        envBlock_weight, envBlock_rotationMatrix = eigsh(envBlock_DM)
        
        """Make the sysBlock weigth array in descending order""" 
        # Sorted Method using Numpy 
        sysBlock_idx = sysBlock_weight.argsort()[::-1]
        sysBlock_weight_sort = sysBlock_weight[sysBlock_idx]
        Isys = sysBlock_idx

        # Check Entanglement 
        # If the resulted array is [1,0,0,.....,0], implying our target state is unentangled
        # If the resulted array is [0.7, 0.2, 0.02, ... 1e-8] , implying our targert state is an enatangled state 

        # print(np.real(sysBlock_weight_sort[:min(len(sysBlock_weight_sort), MaximalStates) ]))

        # von Neumann entropy of sysBlock 
        # locally update the dummy variable SE_local
        SE_local = -  (np.real(sysBlock_weight_sort[:min(MaximalStates, len(sysBlock_rotationMatrix) )]) * np.log(sysBlock_weight_sort[:min(MaximalStates, len(sysBlock_rotationMatrix))] )).sum()


        envBlock_idx = envBlock_weight.argsort()[::-1]
        envBlock_weight_sort = envBlock_weight[envBlock_idx]
        Ienv = envBlock_idx

        #Obtain the truncated basis( There is some bugs in the truncation)
        # sysBlock is a matrix contains eigenvector, but not an array
        sysBlock_rotationMatrix = sysBlock_rotationMatrix[:, Isys[:min(MaximalStates, len(sysBlock_rotationMatrix) ) ]  ]
        envBlock_rotationMatrix = envBlock_rotationMatrix[:, Ienv[:min(MaximalStates, len(envBlock_rotationMatrix) ) ]  ]

        # Step 5: Truncation: 
        # sysBlock 
        sysBlock_Ham = sysBlock_rotationMatrix.conj().T @ sysBlock_Ham.toarray() @ sysBlock_rotationMatrix
        sysBlock_Sigma_x = sysBlock_rotationMatrix.conj().T @ sysBlock_Sigma_x.toarray() @ sysBlock_rotationMatrix
        sysBlock_Sigma_z = sysBlock_rotationMatrix.conj().T @ sysBlock_Sigma_z.toarray() @ sysBlock_rotationMatrix

        # envBlock
        envBlock_Ham = envBlock_rotationMatrix.conj().T @ envBlock_Ham.toarray() @ envBlock_rotationMatrix
        envBlock_Sigma_x = envBlock_rotationMatrix.conj().T @ envBlock_Sigma_x.toarray() @ envBlock_rotationMatrix
        envBlock_Sigma_z = envBlock_rotationMatrix.conj().T @ envBlock_Sigma_z.toarray() @ envBlock_rotationMatrix

        # print("Total length =", sysBlock_Length + envBlock_Length)

    # Append the N=200 values after each while loop 
    SE.append(np.real(SE_local))
    E_GS.append(E_GS_local)

loop_array = np.arange(0,len(E_GS),1)
length_array = np.arange(2,2*sysBlock_Length,2)


# plt.figure(1)
# plt.title(r"Convergence of Ground State Energy ")
# plt.scatter(loop_array, np.array(E_GS) ,facecolors='none', edgecolors='b' ,label=r"DMRG GS" )
# plt.ylabel(r" Energy per Site")
# plt.xlabel(r"Iterations")
# plt.legend()


plt.figure(2)
plt.title(r"Entanglement Spectrum of 1D Tranverse Ising Model")
plt.plot(J_array, np.array(SE))
plt.axhline(y= np.log(2), color='r', linestyle='-')
plt.axvline(x= 1, color='r', linestyle='-', label=r"Critical")

plt.ylabel("Entropy")
plt.xlabel("J/h")

print(SE)
np.save("Entanglement",np.array(SE))
plt.show()