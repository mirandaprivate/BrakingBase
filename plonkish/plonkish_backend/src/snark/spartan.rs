use super::batcheval::batch_eval_proof;
use super::helper::{eR1CSmetadata, SparseRep};
use super::sum_check::{
    batch_sum_check_verifier, first_layer_sum_check, initial_sum_check_verification,
    matrix_eval_sum_check_verifier, par_sum_check_verification, parallel_sum_checks,
};
use crate::pcs::multilinear::brakingbase::{
    Brakingbase, BrakingbaseProverParams, BrakingbaseSpec, BrakingbaseVerifierParams,
};
use crate::pcs::multilinear::brakingbase_helper::point_to_tensor;
use crate::pcs::PolynomialCommitmentScheme;
use crate::piop::GKR::gkr::gkr_verifier;
use crate::poly::Polynomial;
use crate::util::hash::Hash;
use crate::util::transcript::TranscriptRead;
use crate::{poly::multilinear::MultilinearPolynomial, util::transcript::TranscriptWrite};
use ff::PrimeField;
use serde::{de::DeserializeOwned, Serialize};
#[allow(non_snake_case)]
pub fn prove_sat<F, H, S>(
    A: &SparseRep<F>,
    B: &SparseRep<F>,
    C: &SparseRep<F>,
    u: &F,
    z: &MultilinearPolynomial<F>,
    E: &MultilinearPolynomial<F>,
    W: &MultilinearPolynomial<F>,
    metadatas: eR1CSmetadata<F>,
    pp: &BrakingbaseProverParams<F, H>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{

    let fsrp = first_layer_sum_check::<F, H, S>(
        &A,
        &B,
        &C,
        u,
        &z.clone().into_evals(),
        E.clone().into_evals(),
        transcript,
    );

    let rx_basis_evals = point_to_tensor(1, &fsrp).1;

    // let rx_basis_evals = compute_coeff(&first_sum_check_transcript.random_points);
    let par_srp = parallel_sum_checks::<F, H, S>(
        &A,
        &B,
        &C,
        z.clone().into_evals(),
        rx_basis_evals.clone(),
        transcript,
    );
    let w_eval = W.evaluate(&par_srp);
    transcript.write_field_element(&w_eval).unwrap();

    let metadatas = vec![metadatas.A, metadatas.B, metadatas.C];

    batch_eval_proof::<F, H, S>(
        metadatas,
        fsrp,
        par_srp,
        rx_basis_evals,
        E,
        W,
        pp,
        transcript,
    );
}

#[allow(non_snake_case)]
pub fn verify_sat<F, H, S>(
    num_const: usize,
    sparsity: usize,
    vp: &BrakingbaseVerifierParams<F, H>,
    u: F,
    PI: MultilinearPolynomial<F>,
    pi_indices: Vec<usize>,
    transcript: &mut impl TranscriptRead<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let E_commit = transcript.read_commitment().unwrap();
    let W_commit = transcript.read_commitment().unwrap();

    let A_row_commit = transcript.read_commitment().unwrap();
    let A_col_commit = transcript.read_commitment().unwrap();
    let A_val_commit = transcript.read_commitment().unwrap();
    let A_read_ts_row_commit = transcript.read_commitment().unwrap();
    let A_read_ts_col_commit = transcript.read_commitment().unwrap();
    let A_final_ts_row_commit = transcript.read_commitment().unwrap();
    let A_final_ts_col_commit = transcript.read_commitment().unwrap();

    let B_row_commit = transcript.read_commitment().unwrap();
    let B_col_commit = transcript.read_commitment().unwrap();
    let B_val_commit = transcript.read_commitment().unwrap();
    let B_read_ts_row_commit = transcript.read_commitment().unwrap();
    let B_read_ts_col_commit = transcript.read_commitment().unwrap();
    let B_final_ts_row_commit = transcript.read_commitment().unwrap();
    let B_final_ts_col_commit = transcript.read_commitment().unwrap();

    let C_row_commit = transcript.read_commitment().unwrap();
    let C_col_commit = transcript.read_commitment().unwrap();
    let C_val_commit = transcript.read_commitment().unwrap();
    let C_read_ts_row_commit = transcript.read_commitment().unwrap();
    let C_read_ts_col_commit = transcript.read_commitment().unwrap();
    let C_final_ts_row_commit = transcript.read_commitment().unwrap();
    let C_final_ts_col_commit = transcript.read_commitment().unwrap();

    //TODO:- Read commitments;
    // let rx = transcript.first_sum_check_transcript.random_points.clone();
    // let ry = transcript.par_sum_check_transcript.random_points.clone();

    // let Az_claimed_val = transcript.first_sum_check_transcript.Az_claimed_val;
    // let Bz_claimed_val = transcript.first_sum_check_transcript.Bz_claimed_val;
    // let Cz_claimed_val = transcript.first_sum_check_transcript.Cz_claimed_val;
    // let E_final_eval = transcript.E_eval_proof.evaluation;

    let (initial_sc_evals, r_x) =
        initial_sum_check_verification::<F, H, S>(num_const, u, transcript);

    let random_coeffs = transcript.squeeze_challenges(3);

    let par_sc_ic = random_coeffs[0] * initial_sc_evals[0]
        + random_coeffs[1] * initial_sc_evals[1]
        + random_coeffs[2] * initial_sc_evals[2];

    // let eval_point = [rx, ry.clone()].concat();

    let (r_y, a_b_c_claimed, w_eval) = par_sum_check_verification::<F, H, S>(
        num_const,
        pi_indices,
        PI,
        &random_coeffs,
        par_sc_ic,
        transcript,
    );

    let e_rx_commits = transcript.read_commitments(3).unwrap();
    let e_ry_commits = transcript.read_commitments(3).unwrap();

    let random_coeffs = transcript.squeeze_challenges(3);
    let evaluation = random_coeffs
        .iter()
        .zip(a_b_c_claimed.iter())
        .map(|(coeff, eval)| *coeff * *eval)
        .reduce(|acc, g| acc + g)
        .unwrap();

    let (be_sc_rp, be_e_rx_evals, be_e_ry_evals, be_val_evals) =
        matrix_eval_sum_check_verifier::<F, H, S>(
            num_const,
            sparsity,
            evaluation,
            random_coeffs,
            transcript,
        );
    let gamma_tau = transcript.squeeze_challenges(2);

    let (expected_eval, combiners, random_points1, output_layer_eval1) =
        gkr_verifier::<F>(num_const.trailing_zeros() as usize, transcript, 12);

    let (expected_eval, combiners, random_points2, output_layer_eval1) = gkr_verifier::<F>(
        (num_const * sparsity).trailing_zeros() as usize,
        transcript,
        12,
    );

    let rows_evals = transcript.read_field_elements(3).unwrap();
    let cols_evals = transcript.read_field_elements(3).unwrap();
    let read_ts_for_rows_evals = transcript.read_field_elements(3).unwrap();
    let read_ts_for_cols_evals = transcript.read_field_elements(3).unwrap();
    let e_rx_evals = transcript.read_field_elements(3).unwrap();
    let e_ry_evals = transcript.read_field_elements(3).unwrap();
    let final_ts_for_rows_evals = transcript.read_field_elements(3).unwrap();
    let final_ts_for_cols_evals = transcript.read_field_elements(3).unwrap();

    let mut batch_r = Vec::new();
    batch_r.push(&r_x);
    batch_r.push(&r_y);
    batch_r.push(&be_sc_rp);
    batch_r.push(&random_points1);
    batch_r.push(&random_points2);

    let batch_sc_rc = transcript.squeeze_challenges(35);
    let mut initial_claim = batch_sc_rc[0] * initial_sc_evals[3] + batch_sc_rc[1] * w_eval;
    batch_sc_rc
        .iter()
        .skip(2)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * be_e_rx_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(5)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * be_e_ry_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(8)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * be_val_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(11)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * e_rx_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(14)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * e_ry_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(17)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * rows_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(20)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * cols_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(23)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * read_ts_for_rows_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(26)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * read_ts_for_cols_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(29)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * final_ts_for_rows_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(32)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            initial_claim += *coeff * final_ts_for_cols_evals[idx];
        });
    //TODO: Add output layer check
    //TODO: Add input layer check
    batch_sum_check_verifier::<F, H, S>(&batch_r, initial_claim, transcript, &batch_sc_rc);
}

// pub fn evaluate_eq(basis_point: Vec<F>, evaluation_point: Vec<F>) -> F {
//     let mut res = F::ONE;
//     for (x, y) in basis_point.iter().zip(evaluation_point.iter()) {
//         res *= F::ONE - *x - *y + (*x * *y).double()
//     }
//     res
// }

// //...........
// // CODE  for evaluating polynomial at points
// //.............
// pub fn eval(p: &[F], x: F) -> F {
//     // Horner evaluation
//     p.iter().rev().fold(F::ZERO, |acc, &coeff| acc * x + coeff)
// }
