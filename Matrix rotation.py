import numpy as np
A=np.array([[1,2,3],[4,5,6],[7,8,9]])
print("original matrix:")
print(A)
print("\n Rotate 90")
print(np.rot90(A))
print("\n Rotate 180")
print(np.rot90(A,2))
print("\n Rotate 270")
print(np.rot90(A,3))

print("\nFlip left to right:")
print(np.fliplr(A))
print("\nFlip up and down:")
print(np.flipud(A))
print("\nFlip both areas:")
print(np.flip(A))



