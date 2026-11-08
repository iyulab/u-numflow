//! Dense matrix operations.
//!
//! A minimal, row-major dense matrix type with fundamental linear algebra
//! operations required for statistical analysis: multiplication, transpose,
//! determinant, inverse, Cholesky decomposition, and triangular solves.
//!
//! # Design
//!
//! - **Row-major storage**: `data[i * cols + j] = A[i, j]`
//! - **No external dependencies**: Pure Rust, no nalgebra/LAPACK
//! - **Partial pivoting**: Used in LU and Gauss-Jordan for numerical stability
//! - **Explicit error handling**: Returns `Result<T, MatrixError>` with descriptive variants
//!
//! # Examples
//!
//! ```
//! use u_numflow::matrix::Matrix;
//!
//! let a = Matrix::from_rows(&[
//!     &[1.0, 2.0],
//!     &[3.0, 4.0],
//! ]);
//! let b = a.transpose();
//! let c = a.mul_mat(&b).unwrap();
//! assert_eq!(c.rows(), 2);
//! assert_eq!(c.cols(), 2);
//! ```

/// Error type for matrix operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    /// Dimensions do not match for the operation.
    DimensionMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    /// Matrix must be square for this operation.
    NotSquare { rows: usize, cols: usize },
    /// Matrix is singular (zero or near-zero pivot encountered).
    Singular,
    /// Matrix is not symmetric (required for Cholesky).
    NotSymmetric,
    /// Matrix is not positive-definite (required for Cholesky).
    NotPositiveDefinite,
    /// Data length does not match dimensions.
    InvalidData { expected: usize, got: usize },
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "dimension mismatch: expected {}×{}, got {}×{}",
                    expected.0, expected.1, got.0, got.1
                )
            }
            MatrixError::NotSquare { rows, cols } => {
                write!(f, "matrix must be square, got {rows}×{cols}")
            }
            MatrixError::Singular => write!(f, "matrix is singular"),
            MatrixError::NotSymmetric => write!(f, "matrix is not symmetric"),
            MatrixError::NotPositiveDefinite => write!(f, "matrix is not positive-definite"),
            MatrixError::InvalidData { expected, got } => {
                write!(f, "data length mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for MatrixError {}

/// A dense matrix stored in row-major order.
///
/// # Storage
///
/// Elements are stored contiguously: `data[i * cols + j]` holds `A[i, j]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Creates a matrix from raw data in row-major order.
    ///
    /// # Errors
    /// Returns `Err` if `data.len() != rows * cols`.
    ///
    /// # Examples
    /// ```
    /// use u_numflow::matrix::Matrix;
    /// let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// assert_eq!(m.get(0, 2), 3.0);
    /// assert_eq!(m.get(1, 0), 4.0);
    /// ```
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::InvalidData {
                expected: rows * cols,
                got: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    /// Creates a matrix from row slices.
    ///
    /// # Panics
    /// Panics if rows have inconsistent lengths or `rows` is empty.
    ///
    /// # Examples
    /// ```
    /// use u_numflow::matrix::Matrix;
    /// let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    /// assert_eq!(m.get(1, 1), 4.0);
    /// ```
    pub fn from_rows(rows: &[&[f64]]) -> Self {
        assert!(!rows.is_empty(), "must have at least one row");
        let ncols = rows[0].len();
        assert!(ncols > 0, "must have at least one column");
        let nrows = rows.len();
        let mut data = Vec::with_capacity(nrows * ncols);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                ncols,
                "row {i} has {} columns, expected {ncols}",
                row.len()
            );
            data.extend_from_slice(row);
        }
        Self {
            data,
            rows: nrows,
            cols: ncols,
        }
    }

    /// Creates a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates an identity matrix.
    ///
    /// # Examples
    /// ```
    /// use u_numflow::matrix::Matrix;
    /// let eye = Matrix::identity(3);
    /// assert_eq!(eye.get(0, 0), 1.0);
    /// assert_eq!(eye.get(0, 1), 0.0);
    /// assert_eq!(eye.get(2, 2), 1.0);
    /// ```
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Creates a column vector (n×1 matrix) from a slice.
    pub fn from_col(data: &[f64]) -> Self {
        Self {
            data: data.to_vec(),
            rows: data.len(),
            cols: 1,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the element at (row, col).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Sets the element at (row, col).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    /// Returns the raw data as a slice.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns a row as a slice.
    #[inline]
    pub fn row(&self, row: usize) -> &[f64] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Returns the diagonal elements.
    pub fn diag(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    /// Returns true if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    // ========================================================================
    // Basic Operations
    // ========================================================================

    /// Transpose: returns Aᵀ.
    ///
    /// # Examples
    /// ```
    /// use u_numflow::matrix::Matrix;
    /// let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
    /// let t = m.transpose();
    /// assert_eq!(t.rows(), 3);
    /// assert_eq!(t.cols(), 2);
    /// assert_eq!(t.get(0, 1), 4.0);
    /// ```
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    /// Matrix addition: A + B.
    ///
    /// # Errors
    /// Returns `Err` if dimensions do not match.
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Matrix subtraction: A - B.
    ///
    /// # Errors
    /// Returns `Err` if dimensions do not match.
    pub fn sub(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Ok(Self {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Scalar multiplication: c · A.
    pub fn scale(&self, c: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| c * x).collect();
        Self {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Matrix multiplication: A · B.
    ///
    /// Uses i-k-j loop order for better cache locality on row-major storage.
    ///
    /// # Errors
    /// Returns `Err` if `self.cols != other.rows`.
    ///
    /// # Complexity
    /// O(n·m·p) where self is n×m and other is m×p.
    ///
    /// # Examples
    /// ```
    /// use u_numflow::matrix::Matrix;
    /// let a = Matrix::identity(3);
    /// let b = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
    /// let c = a.mul_mat(&b).unwrap();
    /// assert_eq!(c.get(2, 2), 9.0);
    /// ```
    pub fn mul_mat(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (other.rows, other.cols),
            });
        }
        let mut result = Self::zeros(self.rows, other.cols);
        // i-k-j loop order for row-major cache friendliness
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.data[i * self.cols + k];
                let row_start = i * other.cols;
                let other_row_start = k * other.cols;
                for j in 0..other.cols {
                    result.data[row_start + j] += a_ik * other.data[other_row_start + j];
                }
            }
        }
        Ok(result)
    }

    /// Matrix-vector multiplication: A · v.
    ///
    /// # Errors
    /// Returns `Err` if `self.cols != v.len()`.
    pub fn mul_vec(&self, v: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if self.cols != v.len() {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                got: (v.len(), 1),
            });
        }
        let mut result = vec![0.0; self.rows];
        for (i, res) in result.iter_mut().enumerate() {
            let row_start = i * self.cols;
            *res = self.data[row_start..row_start + self.cols]
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| a * b)
                .sum();
        }
        Ok(result)
    }

    /// Frobenius norm: ‖A‖_F = √(Σᵢⱼ aᵢⱼ²).
    ///
    /// Summed relative to the largest entry, so the squares neither underflow
    /// to 0 for entries below 1e-154 nor overflow to infinity above 1e154 --
    /// the tolerances built on the norm then mean the same in any unit.
    pub fn frobenius_norm(&self) -> f64 {
        let peak = self.data.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if peak == 0.0 || !peak.is_finite() {
            // Zero, or an infinity/NaN entry: the plain sum says so too.
            return self.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        }
        peak * self
            .data
            .iter()
            .map(|x| (x / peak) * (x / peak))
            .sum::<f64>()
            .sqrt()
    }

    /// Checks whether the matrix is symmetric within tolerance.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let cols = self.cols;
        for j in 0..cols {
            self.data.swap(a * cols + j, b * cols + j);
        }
    }

    // ========================================================================
    // Decompositions
    // ========================================================================

    /// Determinant via LU decomposition with partial pivoting.
    ///
    /// # Errors
    /// Returns `Err(NotSquare)` if the matrix is not square.
    /// Returns `Err(Singular)` if a zero pivot is encountered.
    ///
    /// # Complexity
    /// O(n³/3).
    ///
    /// # Examples
    /// ```
    /// use u_numflow::matrix::Matrix;
    /// let m = Matrix::from_rows(&[&[2.0, 3.0], &[1.0, 4.0]]);
    /// assert!((m.determinant().unwrap() - 5.0).abs() < 1e-10);
    /// ```
    pub fn determinant(&self) -> Result<f64, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        if n == 0 {
            return Ok(1.0);
        }
        if n == 1 {
            return Ok(self.data[0]);
        }

        let mut work = self.clone();
        let mut sign = 1.0_f64;
        let pivot_tol = 1e-15 * self.frobenius_norm().max(1e-300);

        for k in 0..n {
            // Partial pivoting
            let mut max_val = work.get(k, k).abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = work.get(i, k).abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            if max_val <= pivot_tol {
                return Ok(0.0); // Singular → det = 0
            }
            if max_row != k {
                work.swap_rows(k, max_row);
                sign = -sign;
            }

            let pivot = work.get(k, k);
            for i in (k + 1)..n {
                let factor = work.get(i, k) / pivot;
                for j in (k + 1)..n {
                    let val = work.get(i, j) - factor * work.get(k, j);
                    work.set(i, j, val);
                }
            }
        }

        let mut det = sign;
        for i in 0..n {
            det *= work.get(i, i);
        }
        Ok(det)
    }

    /// Matrix inverse via Gauss-Jordan elimination with partial pivoting.
    ///
    /// # Algorithm
    /// Augments [A | I], reduces to [I | A⁻¹] using row operations.
    ///
    /// Reference: Golub & Van Loan (1996), *Matrix Computations*, §1.2.
    ///
    /// # Errors
    /// Returns `Err(Singular)` if the matrix is singular.
    ///
    /// # Examples
    /// ```
    /// use u_numflow::matrix::Matrix;
    /// let a = Matrix::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]);
    /// let inv = a.inverse().unwrap();
    /// let eye = a.mul_mat(&inv).unwrap();
    /// assert!((eye.get(0, 0) - 1.0).abs() < 1e-10);
    /// assert!(eye.get(0, 1).abs() < 1e-10);
    /// ```
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        if n == 0 {
            return Ok(Self::zeros(0, 0));
        }

        // Augmented matrix [A | I]
        let n2 = 2 * n;
        let mut aug = Self::zeros(n, n2);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, 1.0);
        }

        let pivot_tol = 1e-14 * self.frobenius_norm().max(1e-300);

        for k in 0..n {
            // Partial pivoting
            let mut max_val = aug.get(k, k).abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = aug.get(i, k).abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            if max_val <= pivot_tol {
                return Err(MatrixError::Singular);
            }
            if max_row != k {
                aug.swap_rows(k, max_row);
            }

            // Scale pivot row
            let pivot = aug.get(k, k);
            for j in 0..n2 {
                aug.set(k, j, aug.get(k, j) / pivot);
            }

            // Eliminate column k in all other rows
            for i in 0..n {
                if i != k {
                    let factor = aug.get(i, k);
                    for j in 0..n2 {
                        let val = aug.get(i, j) - factor * aug.get(k, j);
                        aug.set(i, j, val);
                    }
                }
            }
        }

        // Extract right half
        let mut inv = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug.get(i, n + j));
            }
        }
        Ok(inv)
    }

    /// Cholesky decomposition: returns lower-triangular L such that A = L·Lᵀ.
    ///
    /// # Algorithm
    /// Column-by-column Cholesky-Banachiewicz factorization.
    ///
    /// Reference: Golub & Van Loan (1996), *Matrix Computations*, Algorithm 4.2.1.
    ///
    /// # Requirements
    /// Matrix must be symmetric and positive-definite.
    ///
    /// # Complexity
    /// O(n³/3).
    ///
    /// # Examples
    /// ```
    /// use u_numflow::matrix::Matrix;
    /// let a = Matrix::from_rows(&[
    ///     &[4.0, 2.0],
    ///     &[2.0, 3.0],
    /// ]);
    /// let l = a.cholesky().unwrap();
    /// let llt = l.mul_mat(&l.transpose()).unwrap();
    /// assert!((llt.get(0, 0) - 4.0).abs() < 1e-10);
    /// assert!((llt.get(0, 1) - 2.0).abs() < 1e-10);
    /// ```
    pub fn cholesky(&self) -> Result<Self, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        let sym_tol = 1e-10 * self.frobenius_norm().max(1e-300);
        if !self.is_symmetric(sym_tol) {
            return Err(MatrixError::NotSymmetric);
        }

        let mut l = Self::zeros(n, n);

        for j in 0..n {
            // Diagonal entry
            let mut sum = 0.0;
            for k in 0..j {
                let ljk = l.get(j, k);
                sum += ljk * ljk;
            }
            let diag = self.get(j, j) - sum;
            if diag <= 0.0 {
                return Err(MatrixError::NotPositiveDefinite);
            }
            l.set(j, j, diag.sqrt());

            // Below-diagonal entries
            let ljj = l.get(j, j);
            for i in (j + 1)..n {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                l.set(i, j, (self.get(i, j) - sum) / ljj);
            }
        }

        Ok(l)
    }

    /// Solves the linear system A·x = b using Cholesky decomposition.
    ///
    /// Equivalent to computing x = A⁻¹·b but more efficient and stable.
    ///
    /// # Algorithm
    /// 1. Decompose A = L·Lᵀ via Cholesky
    /// 2. Solve L·y = b (forward substitution)
    /// 3. Solve Lᵀ·x = y (backward substitution)
    ///
    /// # Requirements
    /// Matrix must be symmetric positive-definite. `b.len()` must equal `self.rows()`.
    pub fn cholesky_solve(&self, b: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if b.len() != self.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, 1),
                got: (b.len(), 1),
            });
        }
        let l = self.cholesky()?;
        let y = solve_lower_triangular(&l, b)?;
        let lt = l.transpose();
        solve_upper_triangular(&lt, &y)
    }

    /// Eigenvalue decomposition of a real symmetric matrix: the classical
    /// Jacobi rotation algorithm below 64 rows, Householder
    /// tridiagonalisation and implicit QL from there on.
    ///
    /// Returns `(eigenvalues, eigenvectors)` where eigenvalues are sorted
    /// in **descending** order and eigenvectors are the corresponding columns
    /// of the returned matrix (column `i` is the eigenvector for eigenvalue `i`).
    ///
    /// # Algorithm
    ///
    /// Cyclic Jacobi rotations zero off-diagonal elements iteratively.
    /// Converges quadratically for symmetric matrices, and finds small
    /// eigenvalues to high relative accuracy.
    ///
    /// From 64 rows the matrix is reduced to
    /// tridiagonal form by Householder reflections and its eigensystem found
    /// by the implicit QL algorithm (EISPACK `tred2` / `tql2`), a few times
    /// the work of one Jacobi sweep in all. An eigenvector's sign is not
    /// defined: the two algorithms may return opposite signs.
    ///
    /// Reference: Golub & Van Loan (1996), "Matrix Computations", §8.3–8.4
    ///
    /// # Complexity
    ///
    /// Jacobi: O(n³) per sweep, typically 5–10 sweeps. Householder + QL: O(n³)
    /// in all (about 4n³/3 for the reduction, 3n³ with the eigenvectors).
    ///
    /// # Errors
    ///
    /// Returns `NotSquare` if the matrix is not square, or `NotSymmetric`
    /// if the matrix is not symmetric within tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_numflow::matrix::Matrix;
    ///
    /// let a = Matrix::from_rows(&[
    ///     &[4.0, 1.0],
    ///     &[1.0, 3.0],
    /// ]);
    /// let (eigenvalues, eigenvectors) = a.eigen_symmetric().unwrap();
    ///
    /// // Eigenvalues of [[4,1],[1,3]] are (7+√5)/2 ≈ 4.618 and (7-√5)/2 ≈ 2.382
    /// assert!((eigenvalues[0] - 4.618).abs() < 0.01);
    /// assert!((eigenvalues[1] - 2.382).abs() < 0.01);
    ///
    /// // Eigenvectors are orthonormal
    /// let dot: f64 = (0..2).map(|i| eigenvectors.get(i, 0) * eigenvectors.get(i, 1)).sum();
    /// assert!(dot.abs() < 1e-10);
    /// ```
    pub fn eigen_symmetric(&self) -> Result<(Vec<f64>, Matrix), MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        // Solve a copy divided by a power of two near its largest entry -- an
        // exact division, so no digit changes -- and multiply the eigenvalues
        // back: the squares a norm or a rotation forms underflow to 0 for
        // entries below 1e-154 (a covariance in 1e-100 units) and overflow
        // above 1e154, and the answer must not depend on the unit.
        let peak = self.data.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if !(peak.is_finite() && peak > 0.0) {
            return self.eigen_symmetric_scaled();
        }
        let scale = 2.0_f64.powi((peak.log2().floor() as i32).clamp(-1022, 1023));
        let scaled = Matrix {
            data: self.data.iter().map(|x| x / scale).collect(),
            rows: self.rows,
            cols: self.cols,
        };
        let (mut values, vectors) = scaled.eigen_symmetric_scaled()?;
        for value in &mut values {
            *value *= scale;
        }
        Ok((values, vectors))
    }

    /// [`eigen_symmetric`](Self::eigen_symmetric) on a matrix whose largest
    /// entry is of order 1 (or that is zero or not finite).
    fn eigen_symmetric_scaled(&self) -> Result<(Vec<f64>, Matrix), MatrixError> {
        let n = self.rows;
        // Symmetry tolerance: relative to matrix scale
        let sym_tol = 1e-10 * self.frobenius_norm();
        if !self.is_symmetric(sym_tol) {
            return Err(MatrixError::NotSymmetric);
        }

        if n >= HOUSEHOLDER_MIN_ORDER {
            if let Some(result) = self.eigen_symmetric_householder() {
                return Ok(result);
            }
            // QL did not converge (not seen on a finite symmetric matrix):
            // Jacobi below always finishes.
        }

        // Work on a mutable copy of the matrix
        let mut a = self.data.clone();
        // Eigenvector accumulator — starts as identity
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        let max_sweeps = 100;
        // Relative to the matrix: an absolute tolerance would leave a matrix
        // whose entries are all below it unrotated and return its diagonal as
        // the eigenvalues.
        let scale = self.frobenius_norm();
        if scale == 0.0 {
            return Ok((vec![0.0; n], Matrix::identity(n)));
        }
        let tol = 1e-15 * scale;

        for _ in 0..max_sweeps {
            // Compute off-diagonal Frobenius norm
            let mut off_norm = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    off_norm += 2.0 * a[i * n + j] * a[i * n + j];
                }
            }
            off_norm = off_norm.sqrt();

            if off_norm < tol {
                break;
            }

            // One full sweep: rotate each (p, q) pair
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[p * n + q];
                    if apq.abs() < tol * 0.01 {
                        continue;
                    }

                    let app = a[p * n + p];
                    let aqq = a[q * n + q];
                    let diff = aqq - app;

                    // Compute rotation angle
                    let (cos, sin) = if diff.abs() < 1e-300 {
                        // Special case: diagonal elements equal
                        let s = std::f64::consts::FRAC_1_SQRT_2;
                        (s, if apq > 0.0 { s } else { -s })
                    } else {
                        let tau = diff / (2.0 * apq);
                        // t = sign(tau) / (|tau| + sqrt(1 + tau²))
                        let t = if tau >= 0.0 {
                            1.0 / (tau + (1.0 + tau * tau).sqrt())
                        } else {
                            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                        };
                        let c = 1.0 / (1.0 + t * t).sqrt();
                        let s = t * c;
                        (c, s)
                    };

                    // Apply rotation to matrix A (symmetric, only update needed parts)
                    a[p * n + p] -=
                        2.0 * sin * cos * apq + sin * sin * (a[q * n + q] - a[p * n + p]);
                    a[q * n + q] += 2.0 * sin * cos * apq + sin * sin * (aqq - app); // use original aqq, app
                    a[p * n + q] = 0.0;
                    a[q * n + p] = 0.0;

                    // Actually, let's use the standard Jacobi rotation formula properly.
                    // Reset and recompute.
                    // Undo the above:
                    a[p * n + p] = app;
                    a[q * n + q] = aqq;
                    a[p * n + q] = apq;
                    a[q * n + p] = apq;

                    // Standard update: for all rows/cols
                    // First update rows p and q for all columns
                    for r in 0..n {
                        if r == p || r == q {
                            continue;
                        }
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        a[r * n + p] = cos * arp - sin * arq;
                        a[r * n + q] = sin * arp + cos * arq;
                        a[p * n + r] = a[r * n + p]; // symmetric
                        a[q * n + r] = a[r * n + q]; // symmetric
                    }

                    // Update diagonal and off-diagonal (p,q)
                    let new_pp = cos * cos * app - 2.0 * sin * cos * apq + sin * sin * aqq;
                    let new_qq = sin * sin * app + 2.0 * sin * cos * apq + cos * cos * aqq;
                    a[p * n + p] = new_pp;
                    a[q * n + q] = new_qq;
                    a[p * n + q] = 0.0;
                    a[q * n + p] = 0.0;

                    // Accumulate eigenvectors: V = V * J
                    for r in 0..n {
                        let vp = v[r * n + p];
                        let vq = v[r * n + q];
                        v[r * n + p] = cos * vp - sin * vq;
                        v[r * n + q] = sin * vp + cos * vq;
                    }
                }
            }
        }

        // Extract eigenvalues from diagonal
        let mut eigen_pairs: Vec<(f64, Vec<f64>)> = (0..n)
            .map(|i| {
                let eigenvalue = a[i * n + i];
                let eigenvector: Vec<f64> = (0..n).map(|r| v[r * n + i]).collect();
                (eigenvalue, eigenvector)
            })
            .collect();

        // Sort by eigenvalue descending
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(val, _)| *val).collect();
        let mut eigvec_data = vec![0.0; n * n];
        for (col, (_, vec)) in eigen_pairs.iter().enumerate() {
            for (row, &val) in vec.iter().enumerate() {
                eigvec_data[row * n + col] = val;
            }
        }
        let eigenvectors = Matrix {
            data: eigvec_data,
            rows: n,
            cols: n,
        };

        Ok((eigenvalues, eigenvectors))
    }

    /// Eigensystem of a symmetric matrix by Householder tridiagonalisation and
    /// the implicit QL algorithm; `None` if QL does not converge or the matrix
    /// holds a non-finite entry. Eigenvalues descending, eigenvectors as columns.
    fn eigen_symmetric_householder(&self) -> Option<(Vec<f64>, Matrix)> {
        let n = self.rows;
        if self.data.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Work on the matrix divided by its largest entry, so nothing over- or
        // underflows at any scale (a sum of squares would); the eigenvalues are
        // multiplied back.
        let scale = self.data.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if scale == 0.0 {
            return Some((vec![0.0; n], Matrix::identity(n)));
        }
        let mut v: Vec<f64> = self.data.iter().map(|x| x / scale).collect();
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];
        // The transformation comes back one column per row of memory, which is
        // what the QL rotations want: each acts on a pair of eigenvector columns.
        householder_tridiagonalise(&mut v, &mut d, &mut e, n);
        let mut vt = v;
        if !tridiagonal_ql(&mut vt, &mut d, &mut e, n) {
            return None;
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));
        let eigenvalues: Vec<f64> = order.iter().map(|&i| d[i] * scale).collect();
        let mut data = vec![0.0; n * n];
        for (col, &i) in order.iter().enumerate() {
            let row_i = &vt[i * n..(i + 1) * n];
            for (row, &x) in row_i.iter().enumerate() {
                data[row * n + col] = x;
            }
        }
        Some((
            eigenvalues,
            Matrix {
                data,
                rows: n,
                cols: n,
            },
        ))
    }
}

/// Solves L·x = b where L is lower-triangular (forward substitution).
fn solve_lower_triangular(l: &Matrix, b: &[f64]) -> Result<Vec<f64>, MatrixError> {
    let n = l.rows();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for (j, &xj) in x[..i].iter().enumerate() {
            sum += l.get(i, j) * xj;
        }
        let diag = l.get(i, i);
        if diag.abs() < 1e-300 {
            return Err(MatrixError::Singular);
        }
        x[i] = (b[i] - sum) / diag;
    }
    Ok(x)
}

/// Solves U·x = b where U is upper-triangular (backward substitution).
fn solve_upper_triangular(u: &Matrix, b: &[f64]) -> Result<Vec<f64>, MatrixError> {
    let n = u.rows();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for (off, &xj) in x[i + 1..].iter().enumerate() {
            sum += u.get(i, i + 1 + off) * xj;
        }
        let diag = u.get(i, i);
        if diag.abs() < 1e-300 {
            return Err(MatrixError::Singular);
        }
        x[i] = (b[i] - sum) / diag;
    }
    Ok(x)
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:>10.4}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

/// Order (rows) from which [`Matrix::eigen_symmetric`] reduces to tridiagonal form
/// instead of rotating: Jacobi costs several sweeps of `n³` and is best for
/// small matrices.
const HOUSEHOLDER_MIN_ORDER: usize = 64;

/// Householder reduction of the symmetric matrix `v` (`n × n`) to tridiagonal
/// form (EISPACK `tred2`, after JAMA): on return `d` is the diagonal, `e[1..]`
/// the subdiagonal (`e[0] = 0`) and `v` the orthogonal matrix of the
/// transformation.
///
/// `v` is stored **column by column** (element `(r, c)` at `c * n + r`): every
/// inner loop of the algorithm runs down a column, which then is a contiguous
/// stretch of memory instead of a stride of `n` -- at 2,000 rows the strided
/// form took three times as long, all of it cache misses. The input is
/// symmetric, so its layout does not matter, and the result is the transform
/// with one column per row of memory, the form the QL step wants.
fn householder_tridiagonalise(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    let at = |r: usize, c: usize| c * n + r;
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
    }
    for i in (1..n).rev() {
        // Scale to avoid under/overflow.
        let mut scale = 0.0;
        let mut h = 0.0;
        for dk in d.iter().take(i) {
            scale += dk.abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
                v[at(j, i)] = 0.0;
            }
        } else {
            // Generate the Householder vector.
            for dk in d.iter_mut().take(i) {
                *dk /= scale;
                h += *dk * *dk;
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for ej in e.iter_mut().take(i) {
                *ej = 0.0;
            }
            // Apply the similarity transformation to the remaining columns.
            for j in 0..i {
                f = d[j];
                v[at(j, i)] = f;
                g = e[j] + v[at(j, j)] * f;
                for k in (j + 1)..i {
                    g += v[at(k, j)] * d[k];
                    e[k] += v[at(k, j)] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[at(k, j)] -= f * e[k] + g * d[k];
                }
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
            }
        }
        d[i] = h;
    }
    // Accumulate the transformations.
    for i in 0..n - 1 {
        v[at(n - 1, i)] = v[at(i, i)];
        v[at(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[at(k, i + 1)] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[at(k, i + 1)] * v[at(k, j)];
                }
                for k in 0..=i {
                    v[at(k, j)] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[at(k, i + 1)] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
        v[at(n - 1, j)] = 0.0;
    }
    v[at(n - 1, n - 1)] = 1.0;
    e[0] = 0.0;
}

/// Implicit QL on the tridiagonal matrix (`d` diagonal, `e[1..]` subdiagonal)
/// with the eigenvectors accumulated in `vt`, **one eigenvector per row**
/// (EISPACK `tql2`, after JAMA). On return `d` holds the eigenvalues
/// (unsorted) and row `i` of `vt` the eigenvector of `d[i]`. `false` if an
/// eigenvalue needs more than 60 iterations.
fn tridiagonal_ql(vt: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) -> bool {
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    let mut f = 0.0;
    let mut tst1: f64 = 0.0;
    let eps = f64::EPSILON;
    for l in 0..n {
        // Find a small subdiagonal element.
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        // m == l: d[l] is an eigenvalue. Otherwise iterate.
        if m > l {
            let mut iterations = 0;
            loop {
                iterations += 1;
                if iterations > 60 {
                    return false;
                }
                // Compute the implicit shift.
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for di in d.iter_mut().take(n).skip(l + 2) {
                    *di -= h;
                }
                f += h;

                // Implicit QL transformation.
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);

                    // Accumulate the rotation in the eigenvector rows i, i+1.
                    let (lo, hi) = vt.split_at_mut((i + 1) * n);
                    let row_i = &mut lo[i * n..];
                    let row_i1 = &mut hi[..n];
                    for (a, b) in row_i.iter_mut().zip(row_i1.iter_mut()) {
                        let hv = *b;
                        *b = s * *a + c * hv;
                        *a = c * *a - s * hv;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                // Converged when the subdiagonal element is negligible.
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Construction ---

    #[test]
    fn test_new_valid() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn test_new_invalid_length() {
        assert!(Matrix::new(2, 3, vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn test_from_rows() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    fn test_zeros() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.get(2, 3), 0.0);
    }

    #[test]
    fn test_identity() {
        let eye = Matrix::identity(3);
        assert_eq!(eye.get(0, 0), 1.0);
        assert_eq!(eye.get(1, 1), 1.0);
        assert_eq!(eye.get(2, 2), 1.0);
        assert_eq!(eye.get(0, 1), 0.0);
        assert_eq!(eye.get(1, 2), 0.0);
    }

    #[test]
    fn test_diag() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        assert_eq!(m.diag(), vec![1.0, 5.0, 9.0]);
    }

    // --- Basic operations ---

    #[test]
    fn test_transpose() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_transpose_twice() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let tt = m.transpose().transpose();
        assert_eq!(m, tt);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = Matrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(1, 1), 12.0);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(3, 2);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_sub() {
        let a = Matrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let b = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let c = a.sub(&b).unwrap();
        assert_eq!(c.get(0, 0), 4.0);
        assert_eq!(c.get(1, 1), 4.0);
    }

    #[test]
    fn test_scale() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let s = m.scale(2.0);
        assert_eq!(s.get(0, 0), 2.0);
        assert_eq!(s.get(1, 1), 8.0);
    }

    // --- Multiplication ---

    #[test]
    fn test_mul_identity() {
        let a = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        let eye = Matrix::identity(3);
        let result = a.mul_mat(&eye).unwrap();
        assert_eq!(a, result);
    }

    #[test]
    fn test_mul_2x2() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = Matrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let c = a.mul_mat(&b).unwrap();
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_mul_nonsquare() {
        let a = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let b = Matrix::from_rows(&[&[7.0, 8.0], &[9.0, 10.0], &[11.0, 12.0]]);
        let c = a.mul_mat(&b).unwrap();
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        assert_eq!(c.get(0, 0), 58.0);
        assert_eq!(c.get(0, 1), 64.0);
    }

    #[test]
    fn test_mul_vec() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let v = vec![5.0, 6.0];
        let result = a.mul_vec(&v).unwrap();
        assert_eq!(result, vec![17.0, 39.0]);
    }

    #[test]
    fn test_mul_dimension_mismatch() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 3);
        assert!(a.mul_mat(&b).is_err());
    }

    // --- Determinant ---

    #[test]
    fn test_det_2x2() {
        let m = Matrix::from_rows(&[&[2.0, 3.0], &[1.0, 4.0]]);
        assert!((m.determinant().unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_3x3() {
        let m = Matrix::from_rows(&[&[6.0, 1.0, 1.0], &[4.0, -2.0, 5.0], &[2.0, 8.0, 7.0]]);
        // det = 6*(-14-40) - 1*(28-10) + 1*(32+4) = -306
        // Actually: 6(-2*7 - 5*8) - 1(4*7 - 5*2) + 1(4*8 - (-2)*2)
        //         = 6(-14 - 40) - 1(28 - 10) + 1(32 + 4)
        //         = 6*(-54) - 18 + 36 = -324 - 18 + 36 = -306
        assert!((m.determinant().unwrap() - (-306.0)).abs() < 1e-8);
    }

    #[test]
    fn test_det_identity() {
        let eye = Matrix::identity(4);
        assert!((eye.determinant().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_singular() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert!(m.determinant().unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_det_not_square() {
        let m = Matrix::zeros(2, 3);
        assert!(m.determinant().is_err());
    }

    // --- Inverse ---

    #[test]
    fn test_inverse_2x2() {
        let a = Matrix::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]);
        let inv = a.inverse().unwrap();
        let eye = a.mul_mat(&inv).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (eye.get(i, j) - expected).abs() < 1e-10,
                    "A·A⁻¹[{i},{j}] = {}, expected {expected}",
                    eye.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_inverse_3x3() {
        let a = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[0.0, 1.0, 4.0], &[5.0, 6.0, 0.0]]);
        let inv = a.inverse().unwrap();
        let eye = a.mul_mat(&inv).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (eye.get(i, j) - expected).abs() < 1e-10,
                    "A·A⁻¹[{i},{j}] = {}",
                    eye.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_inverse_identity() {
        let eye = Matrix::identity(4);
        let inv = eye.inverse().unwrap();
        assert_eq!(eye, inv);
    }

    #[test]
    fn test_inverse_singular() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert!(m.inverse().is_err());
    }

    // --- Cholesky ---

    #[test]
    fn test_cholesky_2x2() {
        let a = Matrix::from_rows(&[&[4.0, 2.0], &[2.0, 3.0]]);
        let l = a.cholesky().unwrap();
        // L should be lower triangular
        assert!(l.get(0, 1).abs() < 1e-15);
        // Verify A = L·Lᵀ
        let llt = l.mul_mat(&l.transpose()).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (llt.get(i, j) - a.get(i, j)).abs() < 1e-10,
                    "LLᵀ[{i},{j}] = {}, expected {}",
                    llt.get(i, j),
                    a.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_cholesky_3x3() {
        let a = Matrix::from_rows(&[&[25.0, 15.0, -5.0], &[15.0, 18.0, 0.0], &[-5.0, 0.0, 11.0]]);
        let l = a.cholesky().unwrap();
        let llt = l.mul_mat(&l.transpose()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (llt.get(i, j) - a.get(i, j)).abs() < 1e-10,
                    "LLᵀ[{i},{j}] = {}, A[{i},{j}] = {}",
                    llt.get(i, j),
                    a.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_cholesky_identity() {
        let eye = Matrix::identity(3);
        let l = eye.cholesky().unwrap();
        assert_eq!(l, eye);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]);
        assert!(matches!(
            a.cholesky(),
            Err(MatrixError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn test_cholesky_not_symmetric() {
        let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert!(matches!(a.cholesky(), Err(MatrixError::NotSymmetric)));
    }

    // --- Cholesky solve ---

    #[test]
    fn test_cholesky_solve() {
        // A = [[4, 2], [2, 3]], b = [1, 2]
        // Solution: x = [-0.125, 0.75]
        let a = Matrix::from_rows(&[&[4.0, 2.0], &[2.0, 3.0]]);
        let b = vec![1.0, 2.0];
        let x = a.cholesky_solve(&b).unwrap();
        // Verify A·x = b
        let ax = a.mul_vec(&x).unwrap();
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Ax[{i}] = {}, b[{i}] = {}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_cholesky_solve_3x3() {
        let a = Matrix::from_rows(&[&[25.0, 15.0, -5.0], &[15.0, 18.0, 0.0], &[-5.0, 0.0, 11.0]]);
        let b = vec![35.0, 33.0, 6.0];
        let x = a.cholesky_solve(&b).unwrap();
        let ax = a.mul_vec(&x).unwrap();
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-10);
        }
    }

    // --- Frobenius norm ---

    #[test]
    fn test_frobenius_norm() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        // sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477
        assert!((m.frobenius_norm() - 30.0_f64.sqrt()).abs() < 1e-10);
    }

    // --- is_symmetric ---

    #[test]
    fn test_is_symmetric() {
        let sym = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[2.0, 5.0, 6.0], &[3.0, 6.0, 9.0]]);
        assert!(sym.is_symmetric(1e-10));

        let asym = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert!(!asym.is_symmetric(1e-10));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn square_matrix(n: usize) -> impl Strategy<Value = Matrix> {
        proptest::collection::vec(-10.0_f64..10.0, n * n)
            .prop_map(move |data| Matrix::new(n, n, data).expect("valid dimensions"))
    }

    fn spd_matrix(n: usize) -> impl Strategy<Value = Matrix> {
        // Generate random matrix A, then form A'A + nI (guaranteed SPD)
        proptest::collection::vec(-5.0_f64..5.0, n * n).prop_map(move |data| {
            let a = Matrix::new(n, n, data).expect("valid dimensions");
            let ata = a.transpose().mul_mat(&a).expect("compatible");
            let eye_scaled = Matrix::identity(n).scale(n as f64);
            ata.add(&eye_scaled).expect("compatible")
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn transpose_involution(m in square_matrix(3)) {
            let m_tt = m.transpose().transpose();
            for i in 0..3 {
                for j in 0..3 {
                    prop_assert!((m.get(i, j) - m_tt.get(i, j)).abs() < 1e-14);
                }
            }
        }

        #[test]
        fn mul_identity_is_identity(m in square_matrix(3)) {
            let eye = Matrix::identity(3);
            let me = m.mul_mat(&eye).unwrap();
            let em = eye.mul_mat(&m).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    prop_assert!((me.get(i, j) - m.get(i, j)).abs() < 1e-10);
                    prop_assert!((em.get(i, j) - m.get(i, j)).abs() < 1e-10);
                }
            }
        }

        #[test]
        fn det_of_product(a in square_matrix(3), b in square_matrix(3)) {
            // det(A·B) = det(A)·det(B)
            let det_a = a.determinant().unwrap();
            let det_b = b.determinant().unwrap();
            let ab = a.mul_mat(&b).unwrap();
            let det_ab = ab.determinant().unwrap();
            let expected = det_a * det_b;
            // Use relative tolerance for large determinants
            let tol = 1e-6 * expected.abs().max(det_ab.abs()).max(1.0);
            prop_assert!(
                (det_ab - expected).abs() < tol,
                "det(AB)={det_ab}, det(A)*det(B)={expected}"
            );
        }

        #[test]
        fn cholesky_roundtrip(a in spd_matrix(3)) {
            let l = a.cholesky().expect("SPD should decompose");
            let llt = l.mul_mat(&l.transpose()).expect("compatible");
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (llt.get(i, j) - a.get(i, j)).abs();
                    let tol = 1e-8 * a.get(i, j).abs().max(1.0);
                    prop_assert!(
                        diff < tol,
                        "LLᵀ[{i},{j}]={}, A[{i},{j}]={}",
                        llt.get(i, j), a.get(i, j)
                    );
                }
            }
        }

        #[test]
        fn cholesky_solve_roundtrip(a in spd_matrix(3), b in proptest::collection::vec(-10.0_f64..10.0, 3)) {
            let x = a.cholesky_solve(&b).expect("SPD solve should work");
            let ax = a.mul_vec(&x).expect("compatible");
            for i in 0..3 {
                let tol = 1e-8 * b[i].abs().max(1.0);
                prop_assert!(
                    (ax[i] - b[i]).abs() < tol,
                    "Ax[{i}]={}, b[{i}]={}",
                    ax[i], b[i]
                );
            }
        }

        #[test]
        fn inverse_roundtrip(a in spd_matrix(3)) {
            // SPD matrices are always invertible
            let inv = a.inverse().expect("SPD invertible");
            let eye = a.mul_mat(&inv).expect("compatible");
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let diff = (eye.get(i, j) - expected).abs();
                    prop_assert!(
                        diff < 1e-6,
                        "A·A⁻¹[{i},{j}]={}, expected {expected}",
                        eye.get(i, j)
                    );
                }
            }
        }
    }

    #[test]
    fn eigen_symmetric_does_not_depend_on_the_scale_of_the_matrix() {
        let base = Matrix::from_rows(&[&[4.0, 1.0, 0.5], &[1.0, 3.0, 0.25], &[0.5, 0.25, 2.0]]);
        let (want, _) = base.eigen_symmetric().expect("symmetric");
        for scale in [
            1e-3, 1e-12, 1e-18, 1e-24, 1e12, 1e-100, 1e-200, 1e100, 1e200,
        ] {
            let scaled = Matrix::from_rows(&[
                &[4.0 * scale, 1.0 * scale, 0.5 * scale],
                &[1.0 * scale, 3.0 * scale, 0.25 * scale],
                &[0.5 * scale, 0.25 * scale, 2.0 * scale],
            ]);
            let (got, vectors) = scaled.eigen_symmetric().expect("symmetric");
            for (g, w) in got.iter().zip(&want) {
                assert!((g / scale - w).abs() < 1e-12, "scale {scale}: {g} vs {w}");
            }
            // Orthonormal eigenvectors at any scale.
            let dot: f64 = (0..3).map(|r| vectors.get(r, 0) * vectors.get(r, 1)).sum();
            assert!(dot.abs() < 1e-10, "scale {scale}: dot {dot}");
        }
        // The zero matrix has zero eigenvalues and the identity for vectors.
        let (zeros, vectors) = Matrix::new(2, 2, vec![0.0; 4])
            .expect("2x2")
            .eigen_symmetric()
            .expect("symmetric");
        assert_eq!(zeros, vec![0.0, 0.0]);
        assert_eq!(vectors.get(0, 0), 1.0);
    }

    // ── Householder + QL (from HOUSEHOLDER_MIN_ORDER rows) ─────────────

    /// Deterministic symmetric matrix of order `n`.
    fn symmetric(n: usize, seed: u64) -> Matrix {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let x = next();
                data[i * n + j] = x;
                data[j * n + i] = x;
            }
        }
        Matrix::new(n, n, data).expect("square")
    }

    /// `max |A v_k − λ_k v_k|` and `max |VᵀV − I|`.
    fn eigen_residuals(a: &Matrix, values: &[f64], vectors: &Matrix) -> (f64, f64) {
        let n = a.rows();
        let mut residual: f64 = 0.0;
        for (k, &lambda) in values.iter().enumerate() {
            let v: Vec<f64> = (0..n).map(|r| vectors.get(r, k)).collect();
            let av = a.mul_vec(&v).expect("dims");
            for (r, &x) in v.iter().enumerate() {
                residual = residual.max((av[r] - lambda * x).abs());
            }
        }
        let mut orthonormal: f64 = 0.0;
        for i in 0..n {
            for j in i..n {
                let dot: f64 = (0..n).map(|r| vectors.get(r, i) * vectors.get(r, j)).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                orthonormal = orthonormal.max((dot - want).abs());
            }
        }
        (residual, orthonormal)
    }

    #[test]
    fn eigen_symmetric_at_the_householder_order_matches_the_known_spectrum() {
        // The second-difference matrix (2 on the diagonal, -1 beside it) has
        // eigenvalues 2 - 2cos(kπ/(n+1)).
        for n in [HOUSEHOLDER_MIN_ORDER, 100] {
            let mut data = vec![0.0; n * n];
            for i in 0..n {
                data[i * n + i] = 2.0;
                if i + 1 < n {
                    data[i * n + i + 1] = -1.0;
                    data[(i + 1) * n + i] = -1.0;
                }
            }
            let a = Matrix::new(n, n, data).expect("square");
            let (values, vectors) = a.eigen_symmetric().expect("symmetric");
            for (k, &got) in values.iter().enumerate() {
                // Descending: the largest is k = n.
                let order = (n - k) as f64;
                let want = 2.0 - 2.0 * (order * std::f64::consts::PI / (n as f64 + 1.0)).cos();
                assert!((got - want).abs() < 1e-12, "n {n} k {k}: {got} vs {want}");
            }
            let (residual, orthonormal) = eigen_residuals(&a, &values, &vectors);
            assert!(residual < 1e-12, "residual {residual}");
            assert!(orthonormal < 1e-12, "orthonormal {orthonormal}");
        }
    }

    #[test]
    fn eigen_symmetric_of_a_dense_matrix_reconstructs_it() {
        for (n, seed) in [(64, 1), (97, 2), (150, 3)] {
            let a = symmetric(n, seed);
            let (values, vectors) = a.eigen_symmetric().expect("symmetric");
            assert!(values.windows(2).all(|w| w[0] >= w[1]), "descending");
            let trace: f64 = (0..n).map(|i| a.get(i, i)).sum();
            assert!((values.iter().sum::<f64>() - trace).abs() < 1e-10);
            let (residual, orthonormal) = eigen_residuals(&a, &values, &vectors);
            assert!(residual < 1e-11, "n {n}: residual {residual}");
            assert!(orthonormal < 1e-11, "n {n}: orthonormal {orthonormal}");
        }
    }

    #[test]
    fn eigen_symmetric_householder_handles_repeated_and_zero_eigenvalues() {
        let n = HOUSEHOLDER_MIN_ORDER + 6;
        let (values, vectors) = Matrix::identity(n).eigen_symmetric().expect("symmetric");
        assert!(values.iter().all(|&x| (x - 1.0).abs() < 1e-14));
        let (_, orthonormal) = eigen_residuals(&Matrix::identity(n), &values, &vectors);
        assert!(orthonormal < 1e-12);

        // A correlation-like matrix with two identical variables: one eigenvalue is 0.
        let mut a = symmetric(n, 9);
        for i in 0..n {
            a.set(n - 1, i, a.get(n - 2, i));
            a.set(i, n - 1, a.get(i, n - 2));
        }
        let gram = a.transpose().mul_mat(&a).expect("dims");
        let (values, vectors) = gram.eigen_symmetric().expect("symmetric");
        assert!(values.last().copied().unwrap_or(1.0).abs() < 1e-9 * values[0]);
        let (residual, _) = eigen_residuals(&gram, &values, &vectors);
        assert!(residual < 1e-9 * values[0].max(1.0), "residual {residual}");
    }

    #[test]
    fn eigen_symmetric_householder_does_not_depend_on_the_scale() {
        let n = HOUSEHOLDER_MIN_ORDER + 3;
        let base = symmetric(n, 5);
        let (want, _) = base.eigen_symmetric().expect("symmetric");
        for scale in [1e-200, 1e-150, 1e150, 1e200] {
            let scaled = base.scale(scale);
            let (got, vectors) = scaled.eigen_symmetric().expect("symmetric");
            for (g, w) in got.iter().zip(&want) {
                assert!((g / scale - w).abs() < 1e-11, "scale {scale}: {g} vs {w}");
            }
            let (_, orthonormal) = eigen_residuals(
                &base,
                &got.iter().map(|g| g / scale).collect::<Vec<_>>(),
                &vectors,
            );
            assert!(orthonormal < 1e-11);
        }
    }

    #[test]
    fn the_norm_and_the_inverse_do_not_depend_on_the_unit() {
        // A 2x2 covariance in units that make its squares under- or overflow.
        for scale in [1e-200, 1e-100, 1e100, 1e200_f64] {
            let m = Matrix::from_rows(&[&[4.0 * scale, 1.0 * scale], &[1.0 * scale, 3.0 * scale]]);
            let norm = m.frobenius_norm();
            assert!(
                (norm / scale - 27.0_f64.sqrt()).abs() < 1e-12,
                "norm at {scale:e}"
            );
            let inv = m.inverse().expect("not singular in any unit");
            // A^-1 = [[3, -1], [-1, 4]] / 11 / scale
            assert!(
                (inv.get(0, 0) * scale - 3.0 / 11.0).abs() < 1e-12,
                "inverse at {scale:e}"
            );
            assert!((inv.get(0, 1) * scale + 1.0 / 11.0).abs() < 1e-12);
            assert!((m.determinant().is_ok() as u8) == 1);
        }
    }
}
