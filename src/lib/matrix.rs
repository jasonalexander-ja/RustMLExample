use rand::{thread_rng, Rng};


#[derive(Clone)]
pub struct Matrix {
	pub rows: usize,
	pub cols: usize,
	pub data: Vec<Vec<f64>>,
}

impl Matrix {
	pub fn zeros(rows: usize, cols: usize) -> Matrix {
		Matrix {
			rows,
			cols,
			data: vec![vec![0.0; cols]; rows]
		}
	}

	pub fn random(rows: usize, cols: usize) -> Matrix {
		let mut rng = thread_rng();
		let data: Vec<Vec<f64>> = (0..rows).map(
			|_| (0..cols).map(
				|_| rng.gen::<f64>() * 2.0 - 1.0
			).collect()
		).collect();
		
		Matrix {
			rows,
			cols,
			data
		}
	}

	pub fn multiply(&mut self, other: &Matrix) -> Matrix {
		if self.cols != other.rows {
			panic!("Attempted to multiply matrix of {} cols, by matrix of {} rows.", self.cols, other.rows);
		}

		let mut res = Matrix::random(self.rows, other.cols);

		for i in 0..self.rows {
			for j in 0..other.cols {
				let mut sum = 0.0;
				for k in 0..self.cols {
					sum += self.data[i][k] * other.data[k][j];
				}

				res.data[i][j] = sum;
			}
		}

		res
	}

	
	pub fn add(&mut self, other: &Matrix) -> Matrix {
		if self.cols != other.cols || self.rows != other.rows {
			panic!("Attempted to add a matrix of {}{} cols/rows, to a matrix of {}/{} cols/rows.", self.cols, self.rows, other.cols, other.rows);
		}

		let data: Vec<Vec<f64>> = self.data.iter().enumerate().map(
			|(i, r)| r.iter().enumerate().map(move |(j, v)|
				v + other.data[i][j]
			).collect()
		).collect();

		Matrix {
			rows: other.rows,
			cols: self.cols,
			data
		}
	}

	pub fn dot_multiply(&mut self, other: &Matrix) -> Matrix {
		if self.cols != other.cols || self.rows != other.rows {
			panic!("Attempted to dot multiply a matrix of {}/{} cols/rows, by a matrix of {}/{} cols/rows.", self.cols, self.rows, other.cols, other.rows);
		}

		let data: Vec<Vec<f64>> = self.data.iter().enumerate().map(
			|(i, r)| r.iter().enumerate().map(move |(j, v)|
				v * other.data[i][j]
			).collect()
		).collect();

		Matrix {
			rows: other.rows,
			cols: self.cols,
			data
		}
	}

	
	pub fn subtract(&mut self, other: &Matrix) -> Matrix {
		if self.cols != other.cols || self.rows != other.rows {
			panic!("Attempted to subtract a matrix of {}{} cols/rows, from a matrix of {}/{} cols/rows.", self.cols, self.rows, other.cols, other.rows);
		}

		let data: Vec<Vec<f64>> = self.data.iter().enumerate().map(
			|(i, r)| r.iter().enumerate().map(move |(j, v)|
				v - other.data[i][j]
			).collect()
		).collect();

		Matrix {
			rows: other.rows,
			cols: self.cols,
			data
		}
	}

	pub fn map(&mut self, function: &dyn Fn(f64) -> f64) -> Matrix {
		let data: Vec<Vec<f64>> = self.data.iter().map(
			|r| r.iter().map(|v| function(v.clone())).collect()
		).collect();

		Matrix {
			rows: self.rows,
			cols: self.cols,
			data
		}
	}

	pub fn from(data: Vec<Vec<f64>>) -> Matrix {
		Matrix {
			rows: data.len(),
			cols: data[0].len(),
			data
		}
	}

	pub fn transpose(&mut self) -> Matrix {
		let mut res = Matrix::zeros(self.cols, self.rows);

		for (i, row) in res.data.iter_mut().enumerate() {
			for (j, val) in row.iter_mut().enumerate() {
				*val = self.data[j][i].clone();
			}
		}

		res
	}
}
