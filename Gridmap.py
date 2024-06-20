import numpy as np
from scipy.sparse import dok_matrix
from scipy.ndimage import rotate


class Gridmap:
    def __init__(self):
        self.sparse_matrix = dok_matrix((10000, 10000))
        self.position = (5000.0, 5000.0)

    def travel(self, heading):
        x, y = self.position
        new_x = x + np.cos(np.radians(heading))
        new_y = y + np.sin(np.radians(heading))
        self.position = (new_x, new_y)

        # new box check
        new_x = int(new_x)
        new_y = int(new_y)
        if new_x != int(x) or new_y != int(y):
            self.sparse_matrix[new_x, new_y] += 1

    def scan(self, x_size: int, y_size: int) -> np.ndarray:
        matrix = np.zeros(x_size, y_size)

        x, y = self.position
        x = int(x)
        y = int(y)
        lower_x = x - x_size // 2
        upper_x = x + x_size // 2
        lower_y = y - y_size // 2
        upper_y = y + y_size // 2

        for i in range(lower_x, upper_x):
            for j in range(lower_y, upper_y):
                if (i, j) in self.sparse_matrix:
                    matrix[i - lower_x, j - lower_y] = self.sparse_matrix[i, j]

        return matrix

    def scan_interpolated(self, x_size: int, y_size: int, heading: int) -> np.ndarray:
        matrix = self.scan(x_size, y_size)
        interpolated_matrix = rotate(matrix, heading, reshape=False, order=1)
        return interpolated_matrix

    def to_array(self) -> np.ndarray:
        if len(self.sparse_matrix.keys()) == 0:
            return np.array([[]])

        # Find the bounding box of the non-zero values
        rows, cols = zip(*self.sparse_matrix.keys())
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        # Determine the size of the cropped matrix
        num_rows = max_row - min_row + 1
        num_cols = max_col - min_col + 1

        # Create a dense array of the appropriate size
        matrix = np.zeros((num_rows, num_cols))

        # Fill the dense array with the values from the DOK matrix
        for (i, j), value in self.sparse_matrix.items():
            matrix[i - min_row, j - min_col] = value

        return matrix
