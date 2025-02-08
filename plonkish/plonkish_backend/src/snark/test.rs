#![allow(unused)]
use super::helper::SparseRep;
use crate::pcs::multilinear::brakingbase::{
    Brakingbase, BrakingbaseCommitment, BrakingbaseProverParams, BrakingbaseSpec,
};
use crate::pcs::multilinear::{Basefold, BasefoldExtParams};
use crate::pcs::PolynomialCommitmentScheme;
use crate::snark::helper::eR1CSmetadata;
use crate::snark::spartan::{prove_sat, verify_sat};
use crate::util::code::BrakedownSpec;
use crate::util::goldilocksMont::GoldilocksMont;
use crate::util::hash::Hash;
use crate::util::transcript::{Blake2s256Transcript, InMemoryTranscript};
use crate::{
    poly::multilinear::MultilinearPolynomial,
    snark::helper::{sparse_matrix_multiply, ColumnData},
    util::transcript::TranscriptWrite,
};
use blake2::Blake2s256;
use ff::PrimeField;
use halo2_curves::bn256::Fr;
use rand::{rngs::OsRng, Rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{de::DeserializeOwned, Serialize};
use std::{collections::HashMap, time::Instant};

#[derive(Debug)]
pub struct Five {}
impl BrakingbaseSpec for Five {}
impl BrakedownSpec for Five {
    const LAMBDA: f64 = 100.0;
    const ALPHA: f64 = 0.211;
    const BETA: f64 = 0.097;
    const R: f64 = 1.616;
}

impl BasefoldExtParams for Five {
    fn get_reps() -> usize {
        return 656;
    }

    fn get_rate() -> usize {
        return 3;
    }

    fn get_basecode_rounds() -> usize {
        return 0;
    }
    fn get_rs_basecode() -> bool {
        false // Important. Else basefold commit encodes coefficients, not evaluations.
    }
}

type CommitmentScheme = Basefold<GoldilocksMont, Blake2s256, Five>;
#[test]
pub fn er1cs_test() {
    for var in 11..27 {
        let num_const = 1 << var;
        let num_pi_inputs: usize = 8;
        let num_var = num_const - 1;
        let sparsity: usize = 2;
        println!("var is {:?}", var);

        let mut rng = OsRng;
        assert_eq!(
            sparsity.is_power_of_two(),
            true,
            "sparsity must be a power of 2"
        );
        assert_eq!(
            num_pi_inputs.is_power_of_two(),
            true,
            "num_pi_inputs must be a power of 2"
        );

        let param1 = CommitmentScheme::setup(num_const * sparsity, 1, &mut rng).unwrap();
        let param2 = CommitmentScheme::setup(num_const, 1, &mut rng).unwrap();
        let (pp1, vp1) = CommitmentScheme::trim(&param1, num_const * sparsity, 1).unwrap();
        let (pp2, vp2) = CommitmentScheme::trim(&param2, num_const, 1).unwrap();
        let depth = (num_const as u32 * sparsity as u32).trailing_zeros();
        let (A, B, C, z, E, W, u, PI) = construct_matrices::<GoldilocksMont>(
            sparsity as usize,
            num_const,
            num_var as usize,
            num_pi_inputs,
        );

        let mut transcript = Blake2s256Transcript::new(());
        let start_time = Instant::now();
        let (er1cs_metadata, commit1, commit2) = er1cs_commit::<GoldilocksMont, CommitmentScheme>(
            &A,
            &B,
            &C,
            &E,
            &W,
            &pp1,
            &pp2,
            sparsity,
            &mut transcript,
        );
        let commit_size = transcript.clone().into_proof();
        let size = commit_size.len() as f64 / 1024.0;
        println!("Commit size {} KB", size);

        println!("Commit time {:?}", start_time.elapsed());

        let time = Instant::now();
        prove_sat::<GoldilocksMont, CommitmentScheme>(
            &A,
            &B,
            &C,
            &u,
            &MultilinearPolynomial::new(z),
            &E,
            &W,
            er1cs_metadata,
            &pp1,
            &pp2,
            &commit1,
            &commit2,
            &mut transcript,
        );

        println!("Proof time {:?}", time.elapsed());
        let proof = transcript.into_proof();

        let size = proof.len() as f64 / 1024.0;
        println!("Proof size {} KB", size);

        let mut transcript = Blake2s256Transcript::from_proof((), proof.as_slice());

        let pi_indices: Vec<usize> = (0..1 << 5).collect();

        let time = Instant::now();
        verify_sat::<GoldilocksMont, CommitmentScheme>(
            num_const,
            sparsity,
            &vp1,
            &vp2,
            u,
            MultilinearPolynomial::new(PI),
            pi_indices,
            &mut transcript,
        );
        println!("Verifier time {:?}", time.elapsed());
        println!("--------------------------");
    }
}
#[allow(unused)]
pub fn construct_matrices<F: PrimeField + Serialize + DeserializeOwned>(
    sparsity: usize,
    num_const: usize,
    num_var: usize,
    num_pi: usize,
) -> (
    SparseRep<F>,
    SparseRep<F>,
    SparseRep<F>,
    Vec<F>,
    MultilinearPolynomial<F>,
    MultilinearPolynomial<F>,
    F,
    Vec<F>,
) {
    let mut rng = OsRng;
    let W = vec![F::random(rng); num_const];
    let u = F::random(rng);
    let PI = vec![F::random(rng); num_pi];
    let mut E = vec![F::ZERO; num_const];

    let mut Z = W.clone();
    Z.par_iter_mut()
        .enumerate()
        .take(PI.len())
        .for_each(|(i, W)| *W += PI[i]);

    let mut A: HashMap<usize, Vec<ColumnData<F>>> = HashMap::new();
    let mut B: HashMap<usize, Vec<ColumnData<F>>> = HashMap::new();
    let mut C: HashMap<usize, Vec<ColumnData<F>>> = HashMap::new();
    for i in 0..num_const - 1 {
        let mut rng = OsRng;
        let A_row: Vec<ColumnData<F>> = (0..sparsity)
            .map(|_| ColumnData::new((rng.gen_range(0..num_var - 1)) as usize, F::random(rng)))
            .collect();
        let B_row: Vec<ColumnData<F>> = (0..sparsity)
            .map(|_| ColumnData::new((rng.gen_range(0..num_var - 1)) as usize, F::random(rng)))
            .collect();
        let C_row: Vec<ColumnData<F>> = (0..sparsity)
            .map(|_| ColumnData::new((rng.gen_range(0..num_var - 1)) as usize, F::random(rng)))
            .collect();
        A.insert(i, A_row);
        B.insert(i, B_row);
        C.insert(i, C_row);
    }

    let A = SparseRep::new(A);
    let B = SparseRep::new(B);
    let C = SparseRep::new(C);
    let Az = sparse_matrix_multiply(&A, &Z);
    let Bz = sparse_matrix_multiply(&B, &Z);
    let Cz = sparse_matrix_multiply(&C, &Z);
    E.par_iter_mut()
        .enumerate()
        .for_each(|(i, E)| *E = Az[i] * Bz[i] - u * Cz[i]);
    for idx in 0..Az.len() {
        assert_eq!(Az[idx] * Bz[idx], u * Cz[idx] + E[idx]);
    }

    (
        A,
        B,
        C,
        Z,
        MultilinearPolynomial::new(E),
        MultilinearPolynomial::new(W),
        u,
        PI,
    )
}
pub fn er1cs_commit<'a, F, Pcs>(
    A: &'a SparseRep<F>,
    B: &'a SparseRep<F>,
    C: &'a SparseRep<F>,
    E: &'a Pcs::Polynomial,
    W: &'a Pcs::Polynomial,
    pp1: &'a Pcs::ProverParam,
    pp2: &'a Pcs::ProverParam,
    sparsity: usize,
    transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
) -> (eR1CSmetadata<F>, Vec<Pcs::Commitment>, Vec<Pcs::Commitment>)
where
    F: PrimeField + Serialize + DeserializeOwned,

    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    let A_metadata = &A.get_metadata(sparsity);
    let B_metadata = &B.get_metadata(sparsity);
    let C_metadata = &C.get_metadata(sparsity);

    let E_commit = Pcs::commit_and_write(pp2, &E, transcript).unwrap();

    let W_commit = Pcs::commit_and_write(pp2, &W, transcript).unwrap();

    let (
        A_row_commit,
        A_col_commit,
        A_val_commit,
        A_read_ts_row_commit,
        A_read_ts_col_commit,
        A_final_ts_row_commit,
        A_final_ts_col_commit,
    ) = A_metadata.commit::<Pcs>(pp1, pp2, transcript);
    let (
        B_row_commit,
        B_col_commit,
        B_val_commit,
        B_read_ts_row_commit,
        B_read_ts_col_commit,
        B_final_ts_row_commit,
        B_final_ts_col_commit,
    ) = B_metadata.commit::<Pcs>(pp1, pp2, transcript);
    let (
        C_row_commit,
        C_col_commit,
        C_val_commit,
        C_read_ts_row_commit,
        C_read_ts_col_commit,
        C_final_ts_row_commit,
        C_final_ts_col_commit,
    ) = C_metadata.commit::<Pcs>(pp1, pp2, transcript);
    let commit1 = vec![
        A_row_commit,
        B_row_commit,
        C_row_commit,
        A_col_commit,
        B_col_commit,
        C_col_commit,
        A_val_commit,
        B_val_commit,
        C_val_commit,
        A_read_ts_row_commit,
        B_read_ts_row_commit,
        C_read_ts_row_commit,
        A_read_ts_col_commit,
        B_read_ts_col_commit,
        C_read_ts_col_commit,
    ];
    let commit2 = vec![
        A_final_ts_row_commit,
        B_final_ts_row_commit,
        C_final_ts_row_commit,
        A_final_ts_col_commit,
        B_final_ts_col_commit,
        C_final_ts_col_commit,
        E_commit,
        W_commit,
    ];
    (
        eR1CSmetadata::new(A_metadata.clone(), B_metadata.clone(), C_metadata.clone()),
        commit1,
        commit2,
    )
}
