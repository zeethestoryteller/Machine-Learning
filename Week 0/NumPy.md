**Creating Arrays**
* `np.array([1, 2, 3])`: Converts a list or tuple into an ndarray.
* `np.zeros((d1, d2))`: Generates an array filled with zeros of shape `(d1, d2)`.
* `np.ones((d1, d2))`: Generates an array filled with ones of shape `(d1, d2)`.
* `np.arange(start, stop, step)`: Creates a sequence of numbers with a specified interval.
* `np.linspace(start, stop, num)`: Generates a specified number of evenly spaced values over a designated interval.


**Array Inspection & Manipulation**
* `arr.shape`: Returns a tuple representing the dimensions of the array.
* `arr.dtype`: Returns the data type of the elements inside the array.
* `arr.reshape(new_shape)`: Changes the shape of an array without altering its underlying data.
* `arr.flatten()`: Returns a contiguous 1D copy of a multi-dimensional array.


**Mathematical Operations**
* `np.sum(arr)`: Calculates the sum of all elements in the array along a specified axis.
* `np.mean(arr)`: Computes the arithmetic mean of array elements.
* `np.dot(a, b)` or `a @ b`: Performs matrix multiplication between two arrays.
* `np.sqrt(arr)`: Computes the non-negative square root element-wise.


**Statistical & Aggregation Operations**
* `np.median(arr)`: Computes the median of array elements along the specified axis.
* `np.std(arr)`: Calculates the standard deviation, a measure of variation or dispersion.
* `np.var(arr)`: Computes the statistical variance of elements.
* `np.min(arr)` / `np.max(arr)`: Finds the minimum or maximum value within the array.
* `np.argmax(arr)` / `np.argmin(arr)`: Returns the indices of the maximum and minimum values respectively.


**Linear Algebra**
* `np.linalg.inv(arr)`: Computes the mathematical multiplicative inverse of a square matrix.
* `np.linalg.det(arr)`: Calculates the determinant of a square array.
* `np.linalg.eig(arr)`: Computes the eigenvalues and right eigenvectors of a square matrix.


**Indexing, Slicing & Searching**
* `np.where(condition, x, y)`: Returns elements chosen from `x` or `y` depending on whether the conditional expression is true or false.
* `np.unique(arr)`: Finds the distinct, non-repeated elements in an array.
* `np.sort(arr)`: Returns a sorted copy of an array.

