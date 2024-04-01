import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def function_given(t, y):
    return t - y**2

#Q1 Euler's method
def eulers_method(f, a, b, n, y0):
    
    h = (b-a)/n
   
    t, y = a, y0
   
    for i in range(1, n+1):
        y += h * f(t, y)
        t += h
    return y

#Range, iterations, inital point conditions 
a, b = 0, 2
iterations = 10
f0 = 1

approximation = eulers_method(function_given, a, b, iterations, f0)

#Print the approximation to 16 decimal places 
print(f"{approximation:.16f}\n")


#Q2 Runge-Kutta method
def function_given(t, y):
  
    return t - y**2

def runge_kutta(f, a, b, n, y0):
   
    h = (b-a)/n
 
    t, y = a, y0
    
  
    for i in range(1, n+1):
        K_1 = h * function_given(t, y)
        K_2 = h * function_given(t + h/2, y + K_1/2)
        K_3 = h * function_given(t + h/2, y + K_2/2)
        K_4 = h * function_given(t + h, y + K_3)
        
      
        y += (K_1 + 2*K_2 + 2*K_3 + K_4)*(1/6)
        t += h
    return y

#Range, iterations, inital point conditions 
a, b = 0, 2
iterations = 10
f0 = 1
approximation = runge_kutta(function_given, a, b, iterations, f0)

#Print the approximation to 15 decimal places 
print(f"{approximation:.15f}\n")


#Q3 Guassian Elimiation

#Matrix input conditions
mat_a = np.array([[2,-1,1,6],[1,3,1,0],[-1,5,4,-3]]).astype(float)


for i in range(len(mat_a)):
    #Define pivot row
    pivot = i
    for j in range(i+1, len(mat_a)):
        if abs(mat_a[j,i]) > abs(mat_a[pivot,i]):
            pivot = j

    mat_a[[i,pivot]] = mat_a[[pivot,i]]

    for j in range(i+1, len(mat_a)):
        factor = mat_a[j,i]/mat_a[i,i]
        mat_a[j] -= factor * mat_a[i]

#Backwards substitution
x = np.zeros(len(mat_a))
for i in range(len(mat_a)-1,-1,-1):
    x[i] = (mat_a[i,-1] - np.dot(mat_a[i,:-1], x)) / mat_a[i,i]

# Print matrix
print(f"{x}\n")

#Q4 LU Factoriaztion
def LU_Factoriaztion(mat_a):
    #Create L and U matrix
    lfactor = np.zeros_like(mat_a)
    ufactor = np.zeros_like(mat_a)
    nfactor = np.size(mat_a, 0)


    for k in range(nfactor):
        #L=1
        lfactor[k, k] = 1
        #diagonal
        ufactor[k, k] = (mat_a[k, k] - np.dot(lfactor[k, :k], ufactor[:k, k])) / lfactor[k, k]
        #Above diagonal
        for j in range(k+1, nfactor):
            ufactor[k, j] = (mat_a[k, j] - np.dot(lfactor[k, :k], ufactor[:k, j])) / lfactor[k, k]
        #Below diagonal
        for i in range(k+1, nfactor):
            lfactor[i, k] = (mat_a[i, k] - np.dot(lfactor[i, :k], ufactor[:k, k])) / ufactor[k, k]


    return lfactor, ufactor

#populate matrix using values from Q4
mat_a = np.array([[1, 1, 0, 3],[2, 1, -1, 1],[3, -1, -1, 2],[-1, 2, 3, -1]]).astype(float)

#Call L and U matrices
lfactor, ufactor = LU_Factoriaztion(mat_a)

print(lfactor, "\n")
print(ufactor, "\n")


#Q5 Diagonally dominate

#populate matrix
mat_a = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1],[0, 1, 7, 2, 3],[4, 2, 3, 12, 2],[3, 2, 4, 0, 8]])


diagonal_dominant = True
for i in range(mat_a.shape[0]):
    
    sum = np.sum(np.abs(mat_a[i,:])) - np.abs(mat_a[i,i])
    
    #True-false condition. Checking if the matrix is or is not diagonally dominant. 
    if np.abs(mat_a[i,i]) < sum:
        diagonal_dominant = False
        break

if diagonal_dominant:
    print("True")
else:
    print("False\n")


 
#Q6 Positive definite


#populate matrix
mat_a = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

#Checking matrix for positive definite 
positive_definite = True
for k in range(1, mat_a.shape[0]+1):
    minor = mat_a[:k, :k]
    if np.linalg.det(minor) <= 0:

        positive_definite = False
    

if positive_definite:
    print("True\n")
else:
    print("False")

if __name__ == "__main__":
    print()