use crate::pcs::multilinear::brakingbase::{Brakingbase,BrakingbaseProverParams, BrakingbaseSpec};
use crate::util::hash::Hash;
use crate::{
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::transcript::TranscriptWrite,
};
use ff::PrimeField;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;
#[derive(Clone, Copy, Debug)]
pub struct ColumnData<F: PrimeField> {
    pub column: usize,
    pub value: F,
}

impl<F: PrimeField> ColumnData<F> {
    pub fn new(column: usize, value: F) -> Self {
        Self { column, value }
    }
}
#[allow(non_snake_case, non_camel_case_types)]
#[derive(Clone)]
pub struct eR1CSmetadata<F: PrimeField + Serialize + DeserializeOwned> {
    pub A: SparseMetaData<F>,
    pub B: SparseMetaData<F>,
    pub C: SparseMetaData<F>,
}

#[allow(non_snake_case)]
impl <F: PrimeField + Serialize + DeserializeOwned>eR1CSmetadata<F> {
    pub fn new(A: SparseMetaData<F>, B: SparseMetaData<F>, C: SparseMetaData<F>) -> Self {
        Self { A, B, C }
    }
}
#[allow(unused_assignments)]
pub fn get_tuples<F: PrimeField + Serialize + DeserializeOwned>(
    sparse_representation: &SparseRep<F>,
    n_cols: usize,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let sparsity = sparse_representation.sparsity().next_power_of_two();
    let rows = sparse_representation.fourcoeffs.len();
    let cols = n_cols;

    let tuples: Vec<(F, F, F)> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let entries = sparse_representation.fourcoeffs.get(&row).unwrap();
            entries
                .iter()
                .map(|entry| {
                    (
                        F::from(row as u64),
                        F::from(entry.column as u64),
                        entry.value,
                    )
                })
                .collect::<Vec<(F, F, F)>>()
        })
        .flatten()
        .collect();

    let row_col_max = if rows > cols { rows } else { cols };
    let mut row = vec![F::from(row_col_max as u64); sparsity];
    let mut col = vec![F::ZERO; sparsity];
    let mut val = vec![F::ZERO; sparsity];

    (&mut row, &mut col, &mut val, tuples)
        .into_par_iter()
        .for_each(|(row, col, val, tuple)| {
            *row = tuple.0;
            *col = tuple.1;
            *val = tuple.2;
        });

    (row, col, val)
}

pub fn get_timestamps<F: PrimeField>(
    row: &Vec<F>,
    col: &Vec<F>,
    length: usize,
    dim: usize,
) -> TimeStamps<F> {
    let mut read_ts_row = vec![F::ZERO; length];
    let mut read_ts_col = vec![F::ZERO; length];

    let mut final_ts_row = vec![F::ZERO; dim];
    let mut final_ts_col = vec![F::ZERO; dim];

    for i in 0..length {
        let mut bytes = [0; size_of::<u32>()];
        bytes.copy_from_slice(&row[i].to_repr().as_ref()[..size_of::<u32>()]);
        let row_idx = u32::from_le_bytes(bytes) as usize;

        read_ts_row[i] = final_ts_row[row_idx];
        final_ts_row[row_idx] += F::ONE;

        let mut bytes = [0; size_of::<u32>()];
        bytes.copy_from_slice(&col[i].to_repr().as_ref()[..size_of::<u32>()]);
        let col_idx = u32::from_le_bytes(bytes) as usize;

        read_ts_col[i] = final_ts_col[col_idx];
        final_ts_col[col_idx] += F::ONE;
    }

    TimeStamps::new(
        MultilinearPolynomial::new(read_ts_row),
        MultilinearPolynomial::new(read_ts_col),
        MultilinearPolynomial::new(final_ts_row),
        MultilinearPolynomial::new(final_ts_col),
    )
}

#[derive(Clone)]
pub struct TimeStamps<F: PrimeField> {
    pub read_ts_row: MultilinearPolynomial<F>,
    pub read_ts_col: MultilinearPolynomial<F>,
    pub final_ts_row: MultilinearPolynomial<F>,
    pub final_ts_col: MultilinearPolynomial<F>,
}

impl<F: PrimeField> TimeStamps<F> {
    pub fn new(
        read_ts_row: MultilinearPolynomial<F>,
        read_ts_col: MultilinearPolynomial<F>,
        final_ts_row: MultilinearPolynomial<F>,
        final_ts_col: MultilinearPolynomial<F>,
    ) -> Self {
        Self {
            read_ts_row,
            read_ts_col,
            final_ts_row,
            final_ts_col,
        }
    }
}

#[derive(Clone)]
pub struct SparseMetaData<F: PrimeField + Serialize + DeserializeOwned> {
    pub row: MultilinearPolynomial<F>,
    pub col: MultilinearPolynomial<F>,
    pub val: MultilinearPolynomial<F>,
    pub timestamps: TimeStamps<F>,
}

impl<F: PrimeField + Serialize + DeserializeOwned> SparseMetaData<F> {
    pub fn generate(sparse_representation: &SparseRep<F>, n_cols: usize) -> Self {
        let (row, col, val) = get_tuples(sparse_representation, n_cols);
        let timestamps = get_timestamps(&row, &col, val.len(), sparse_representation.dim(n_cols));
        Self {
            row: MultilinearPolynomial::new(row),
            col: MultilinearPolynomial::new(col),
            val: MultilinearPolynomial::new(val),
            timestamps,
        }
    }
    pub fn commit<H: Hash, S: BrakingbaseSpec>(
        &self,
        pp: &BrakingbaseProverParams<F, H>,
        transcript: &mut impl TranscriptWrite<
            <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
            F,
        >,
    ) {
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp, &self.row, transcript,
        )
        .unwrap();
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp, &self.col, transcript,
        )
        .unwrap();
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp, &self.val, transcript,
        )
        .unwrap();
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp,
            &self.timestamps.read_ts_row,
            transcript,
        )
        .unwrap();
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp,
            &self.timestamps.read_ts_col,
            transcript,
        )
        .unwrap();

        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp,
            &self.timestamps.final_ts_row,
            transcript,
        )
        .unwrap();
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp,
            &self.timestamps.final_ts_col,
            transcript,
        )
        .unwrap();
    }
}

#[derive(Debug)]
pub struct SparseRep<F: PrimeField + Serialize + DeserializeOwned> {
    pub fourcoeffs: HashMap<usize, Vec<ColumnData<F>>>,
}

impl<F: PrimeField + Serialize + DeserializeOwned> SparseRep<F> {
    pub fn new(fourcoeffs: HashMap<usize, Vec<ColumnData<F>>>) -> Self {
        Self { fourcoeffs }
    }

    pub fn dim(&self, n_cols: usize) -> usize {
        let rows = self.fourcoeffs.len();
        let cols = n_cols;
        let row_col_max = if rows > cols { rows } else { cols };
        row_col_max.next_power_of_two()
    }
    pub fn sparsity(&self) -> usize {
        let mut nonzero_entries = 0;
        for i in 0..self.fourcoeffs.len() {
            nonzero_entries += self.fourcoeffs.get(&i).unwrap().len()
        }
        nonzero_entries
    }

    pub fn evaluate(self, basis_evals: &Vec<F>, dim: usize) -> F {
        self.fourcoeffs
            .into_par_iter()
            //Iterate over keys in hashmap, which correspond to row indices of the matrix representation of the sparse polynomial.
            .flat_map(
                |(row_idx, row_entries)|

            //For each row, iterate over entries and multiply with corresponding lagrange basis element and flatten returned results
            //into a vector of Fs.
            row_entries.iter().map(|coldata|
            
            basis_evals[(row_idx<<dim) + coldata.column] * coldata.value
        
         ).collect::<Vec<F>>(), //Sum over all returned values to get final evaluation value as an inner product.
            )
            .reduce(|| F::ZERO, |acc, e| acc + e)
    }

    pub fn get_metadata(&self, n_cols: usize) -> SparseMetaData<F> {
        SparseMetaData::generate(&self, n_cols)
    }
    pub fn bind_row_variable(&self, basis_evals: &Vec<F>, n_cols: usize) -> Vec<F> {
        let mut result = vec![F::ZERO; self.dim(n_cols)];
        self.fourcoeffs.iter().for_each(|(row_idx, row_entries)| {
            row_entries
                .iter()
                .for_each(|coldata| result[coldata.column] += basis_evals[*row_idx] * coldata.value)
        });

        result
    }
}

#[allow(unused_variables)]
pub fn sparse_matrix_multiply<F: PrimeField + Serialize + DeserializeOwned>(
    mat: &SparseRep<F>,
    z: &Vec<F>,
) -> Vec<F> {
    let number_of_rows = mat.fourcoeffs.len();
    let mut a_z_vec: Vec<F> = vec![F::ZERO; z.len()];
    for i in 0..number_of_rows {
        match mat.fourcoeffs.get(&i) {
            Some(a_column_data_vec) => {
                let a_column_data_vec = mat.fourcoeffs.get(&i).unwrap();
                let mut a_z = F::ZERO;
                for column_data in a_column_data_vec.iter() {
                    let variable = z[column_data.column];
                    a_z += variable * column_data.value;
                }
                a_z_vec[i] = a_z;
            }
            None => {
                println!("Not found");
                panic!();
            }
        }
    }
    a_z_vec
}
