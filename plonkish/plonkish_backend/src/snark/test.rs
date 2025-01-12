#![allow(unused)]
use super::helper::SparseRep;
use crate::pcs::multilinear::brakingbase::{Brakingbase, BrakingbaseProverParams, BrakingbaseSpec};
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

type Pcs = Brakingbase<GoldilocksMont, Blake2s256, Five>;
#[test]
pub fn er1cs_test() {
    let num_const = 1 << 5;
    let num_pi_inputs: usize = 8;
    let num_var = num_const - 1;
    let sparsity: usize = 1;

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

    let poly_size = 1 << 11; //No of queries are fixed for variables in range [10, 26]

    let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();
    let (pp, vp) = Pcs::trim(&param, poly_size, 1).unwrap();
    let depth = (num_const as u32 * sparsity as u32).trailing_zeros();
    let (A, B, C, z, E, W, u, PI) = construct_matrices::<GoldilocksMont>(
        sparsity as usize,
        num_const,
        num_var as usize,
        num_pi_inputs,
    );

    let mut transcript = Blake2s256Transcript::new(());

    let er1cs_metadata = er1cs_commit::<GoldilocksMont, Blake2s256, Five>(
        &A,
        &B,
        &C,
        &E,
        &W,
        &pp,
        sparsity,
        &mut transcript,
    );

    let time = Instant::now();
    prove_sat::<GoldilocksMont, Blake2s256, Five>(
        &A,
        &B,
        &C,
        &u,
        &MultilinearPolynomial::new(z),
        &E,
        &W,
        er1cs_metadata,
        &pp,
        &mut transcript,
    );

    let proof = transcript.into_proof();
    println!("Time to generate er1cs proof is {:?}", time.elapsed());

    let size = proof.len() as f64 / 1024.0;
    println!("Proof size {}KB", size);

    let mut transcript = Blake2s256Transcript::from_proof((), proof.as_slice());

    let pi_indices: Vec<usize> = (0..1 << 5).collect();

    verify_sat::<GoldilocksMont, Blake2s256, Five>(
        num_const,
        sparsity,
        &vp,
        u,
        MultilinearPolynomial::new(PI),
        pi_indices,
        &mut transcript,
    );
    println!("Time to verify er1cs proof is {:?}", time.elapsed());
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

    //TODO:- Check if Z is correct
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
pub fn er1cs_commit<F: PrimeField + Serialize + DeserializeOwned, H: Hash, S: BrakingbaseSpec>(
    A: &SparseRep<F>,
    B: &SparseRep<F>,
    C: &SparseRep<F>,
    E: &MultilinearPolynomial<F>,
    W: &MultilinearPolynomial<F>,
    pp: &BrakingbaseProverParams<F, H>,
    sparsity: usize,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> eR1CSmetadata<F> {
    let A_metadata = &A.get_metadata(sparsity);
    let B_metadata = &B.get_metadata(sparsity);
    let C_metadata = &C.get_metadata(sparsity);
    let start_time = Instant::now();

    <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(pp, &E, transcript)
        .unwrap();
    <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(pp, &W, transcript)
        .unwrap();
    A_metadata.commit::<H, S>(pp, transcript);
    B_metadata.commit::<H, S>(pp, transcript);
    C_metadata.commit::<H, S>(pp, transcript);

    println!("er1cs commit time {:?}", start_time.elapsed());
    eR1CSmetadata::new(A_metadata.clone(), B_metadata.clone(), C_metadata.clone())
}

// pub fn main(
//     A: SparseRep,
//     B: SparseRep,
//     C: SparseRep,
//     Z: Vec<F>,
//     E: Vec<F>,
//     W: Vec<F>,
//     u: F,
//     PI: Vec<F>,
//     srs: &SRS<BlsCurve>,
//     ver_key: &VerificationKey<BlsCurve>,
//     num_cols: usize,
// ) {
//     let (er1cs_metadata, er1cs_commitments) = er1cs_commit(
//         A.clone(),
//         B.clone(),
//         C.clone(),
//         E.clone(),
//         W.clone(),
//         srs,
//         num_cols,
//     );
//     let mut channel = Channel::initialize_with_affine_point(
//         [
//             er1cs_commitments.E.commitment.to_affine(),
//             er1cs_commitments.W.commitment.to_affine(),
//         ]
//         .as_ref(),
//     );

//     let time = Instant::now();
//     let er1cs_transcript = prove_sat(
//         A,
//         B,
//         C,
//         &u,
//         &MultPolynomial::new(Z),
//         &MultPolynomial::new(E),
//         &MultPolynomial::new(W),
//         er1cs_metadata,
//         srs,
//         &mut channel,
//     );
//     println!("Time to generate er1cs proof is {:?}", time.elapsed());
//     println!(
//         "Proof size {:?}",
//         er1cs_transcript.to_bytes().len() as f64 / 1024f64
//     );
//     let mut channel = Channel::initialize_with_affine_point(
//         [
//             er1cs_commitments.E.commitment.to_affine(),
//             er1cs_commitments.W.commitment.to_affine(),
//         ]
//         .as_ref(),
//     );

//     let pi_indices: Vec<usize> = (0..1 << 5).collect();
//     let time = Instant::now();
//     verify_sat(
//         er1cs_transcript,
//         er1cs_commitments,
//         u,
//         MultPolynomial::new(PI),
//         pi_indices,
//         ver_key,
//         &mut channel,
//     );
//     println!("Time to verify er1cs proof is {:?}", time.elapsed());
// }
