# gram_schmidt
Gram Schmidt Orthogonalization on CPU/GPU

# compile
nvcc -O3 -std=c++11 -D_GPU_ main.cu

# performance on V100
Enter the vector length 10

Enter the # of vectors 1000

==========CPU start==========

11676 usec

==========GPU start==========

8691 usec


Enter the vector length 1000

Enter the # of vectors 100

==========CPU start==========

18185 usec

==========GPU start==========

823 usec
