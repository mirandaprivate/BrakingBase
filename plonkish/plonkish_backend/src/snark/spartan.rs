use super::batcheval::batch_eval_proof;
use super::helper::{eR1CSmetadata, SparseRep};
use super::sum_check::{
    batch_sum_check_verifier, first_layer_sum_check, initial_sum_check_verification,
    matrix_eval_sum_check_verifier, par_sum_check_verification, parallel_sum_checks,
};
use crate::pcs::multilinear::brakingbase::{
    Brakingbase, BrakingbaseCommitment, BrakingbaseProverParams, BrakingbaseSpec,
    BrakingbaseVerifierParams,
};
use crate::pcs::multilinear::brakingbase_helper::{evaluate_eq, point_to_tensor};
use crate::pcs::{Evaluation, PolynomialCommitmentScheme};
use crate::piop::GKR::gkr::gkr_verifier;
use crate::piop::GKR::helper::evaluate_indicies;
use crate::poly::Polynomial;
use crate::util::hash::Hash;
use crate::util::transcript::TranscriptRead;
use crate::{poly::multilinear::MultilinearPolynomial, util::transcript::TranscriptWrite};
use ff::PrimeField;
use itertools::{chain, Itertools};
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
    commit1: &Vec<BrakingbaseCommitment<F, H>>,
    commit2: &Vec<BrakingbaseCommitment<F, H>>,
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
        commit1,
        commit2,
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

    let (initial_sc_evals, r_x) =
        initial_sum_check_verification::<F, H, S>(num_const, u, transcript);

    let random_coeffs = transcript.squeeze_challenges(3);

    let par_sc_ic = random_coeffs[0] * initial_sc_evals[0]
        + random_coeffs[1] * initial_sc_evals[1]
        + random_coeffs[2] * initial_sc_evals[2];

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
    let commit1: Vec<BrakingbaseCommitment<F, H>> = vec![
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
    ]
    .into_iter()
    .chain(e_rx_commits.into_iter())
    .chain(e_ry_commits.into_iter())
    .map(|commit| BrakingbaseCommitment::from_root(commit))
    .collect();

    let commit2: Vec<BrakingbaseCommitment<F, H>> = vec![
        A_final_ts_row_commit,
        B_final_ts_row_commit,
        C_final_ts_row_commit,
        A_final_ts_col_commit,
        B_final_ts_col_commit,
        C_final_ts_col_commit,
        E_commit,
        W_commit,
    ]
    .into_iter()
    .map(|commit| BrakingbaseCommitment::from_root(commit))
    .collect();

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

    let (expected_eval1, combiners1, random_points1, output_layer_eval1) =
        gkr_verifier::<F>(num_const.trailing_zeros() as usize, transcript, 8);

    let (expected_eval2, combiners2, random_points2, output_layer_eval2) = gkr_verifier::<F>(
        (num_const * sparsity).trailing_zeros() as usize,
        transcript,
        12,
    );
    (0..3).for_each(|idx| {
        assert_eq!(
            output_layer_eval1[0]
                * output_layer_eval1[1]
                * output_layer_eval2[2 * idx]
                * output_layer_eval2[2 * idx + 1],
            output_layer_eval1[2 * idx + 2]
                * output_layer_eval1[2 * idx + 3]
                * output_layer_eval2[2 * idx + 6]
                * output_layer_eval2[2 * idx + 7],
            "output layer check failed for row at index {}",
            idx
        );

        assert_eq!(
            output_layer_eval1[8]
                * output_layer_eval1[9]
                * output_layer_eval2[2 * idx + 12]
                * output_layer_eval2[2 * idx + 13],
            output_layer_eval1[2 * idx + 10]
                * output_layer_eval1[2 * idx + 11]
                * output_layer_eval2[2 * idx + 18]
                * output_layer_eval2[2 * idx + 19],
            "output layer check failed for col at index {}",
            idx
        );
    });

    let rows_evals = transcript.read_field_elements(3).unwrap();
    let cols_evals = transcript.read_field_elements(3).unwrap();
    let read_ts_for_rows_evals = transcript.read_field_elements(3).unwrap();
    let read_ts_for_cols_evals = transcript.read_field_elements(3).unwrap();
    let e_rx_evals = transcript.read_field_elements(3).unwrap();
    let e_ry_evals = transcript.read_field_elements(3).unwrap();
    let final_ts_for_rows_evals = transcript.read_field_elements(3).unwrap();
    let final_ts_for_cols_evals = transcript.read_field_elements(3).unwrap();

    input_layer_check1(
        &gamma_tau,
        &r_x,
        &r_y,
        &combiners1,
        &random_points1,
        expected_eval1,
        8,
        &final_ts_for_rows_evals,
        &final_ts_for_cols_evals,
    );

    let input_layer_evaluations = rows_evals
        .iter()
        .chain(e_rx_evals.iter())
        .chain(read_ts_for_rows_evals.iter())
        .chain(cols_evals.iter())
        .chain(e_ry_evals.iter())
        .chain(read_ts_for_cols_evals.iter())
        .collect::<Vec<&F>>();

    input_layer_check2(
        &gamma_tau,
        expected_eval2,
        &combiners2,
        12,
        &input_layer_evaluations,
    );

    let num_var_witness = r_x.len();

    let mut batch_r = Vec::new();
    batch_r.push(r_x);
    batch_r.push(r_y);
    batch_r.push(be_sc_rp);
    batch_r.push(random_points2);
    batch_r.push(random_points1);

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

    let (
        rows_evals,
        cols_evals,
        read_ts_rows_evals,
        read_ts_cols_evals,
        e_rx_evals,
        e_ry_evals,
        final_ts_rows_evals,
        final_ts_cols_evals,
        val_evals,
        E_eval,
        W_eval,
        bt_sc_rp,
    ) = batch_sum_check_verifier::<F, H, S>(batch_r, initial_claim, transcript, &batch_sc_rc);

    // let evals1: Vec<F> = rows_evals
    //     .iter()
    //     .chain(cols_evals.iter())
    //     .chain(val_evals.iter())
    //     .chain(read_ts_rows_evals.iter())
    //     .chain(read_ts_cols_evals.iter())
    //     .chain(e_rx_evals.iter())
    //     .chain(e_ry_evals.iter())
    //     .cloned()
    //     .collect();

    // let evals = chain![
    //     (0..evals1.len()).map(|point| (0, point)),
    //     (0..evals1.len()).map(|poly| (poly, 0)),
    // ]
    // .unique()
    // .collect_vec();
    // let evals1 = evals
    //     .iter()
    //     .copied()
    //     .map(|(poly, point)| Evaluation::new(poly, point, evals1[poly]))
    //     .collect_vec();
    // let evals2: Vec<F> = final_ts_rows_evals
    //     .iter()
    //     .chain(final_ts_cols_evals.iter())
    //     .chain([E_eval].iter())
    //     .chain([W_eval].iter())
    //     .cloned()
    //     .collect();
    // let evals = chain![
    //     (0..evals2.len()).map(|point| (0, point)),
    //     (0..evals2.len()).map(|poly| (poly, 0)),
    // ]
    // .unique()
    // .collect_vec();
    // let evals2 = evals
    //     .iter()
    //     .copied()
    //     .map(|(poly, point)| Evaluation::new(poly, point, evals2[poly]))
    //     .collect_vec();

    let evals1: Vec<F> = rows_evals
        .iter()
        .chain(cols_evals.iter())
        .chain(val_evals.iter())
        .chain(read_ts_rows_evals.iter())
        .chain(read_ts_cols_evals.iter())
        .chain(e_rx_evals.iter())
        .chain(e_ry_evals.iter())
        .cloned()
        .collect();

    let evals1 = evals1
        .iter()
        .map(|eval| Evaluation::new(0, 0, *eval))
        .collect_vec();

    let evals2: Vec<F> = final_ts_rows_evals
        .iter()
        .chain(final_ts_cols_evals.iter())
        .chain([E_eval].iter())
        .chain([W_eval].iter())
        .cloned()
        .collect();

    let evals2 = evals2
        .iter()
        .map(|eval| Evaluation::new(0, 0, *eval))
        .collect_vec();

    <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::batch_verify(
        vp,
        &commit1,
        &[bt_sc_rp.clone()].to_vec(),
        &evals1,
        transcript,
    )
    .unwrap();
    <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::batch_verify(
        vp,
        &commit2,
        &[bt_sc_rp[bt_sc_rp.len() - num_var_witness..].to_vec()].to_vec(),
        &evals2,
        transcript,
    )
    .unwrap();
}
pub fn input_layer_check1<F: PrimeField + Serialize + DeserializeOwned>(
    gamma_tau: &Vec<F>,
    r_x: &Vec<F>,
    r_y: &Vec<F>,
    combiners: &Vec<F>,
    random_points: &Vec<F>,
    expected_eval: F,
    n_circuits: usize,
    final_ts_for_rows_evals: &Vec<F>,
    final_ts_for_cols_evals: &Vec<F>,
) {
    let mut random_points = random_points.clone();
    let r_x_eval = evaluate_eq::<F>(r_x, &random_points);
    let r_y_eval = evaluate_eq::<F>(r_y, &random_points);
    random_points.reverse();
    let indices_eval = evaluate_indicies::<F>(&random_points);
    let gamma_square = gamma_tau[0].square();
    let mut circuit_evals = vec![F::ZERO; n_circuits];

    circuit_evals[0] = indices_eval + gamma_tau[0] * r_x_eval - gamma_tau[1];
    circuit_evals[4] = indices_eval + gamma_tau[0] * r_y_eval - gamma_tau[1];

    (0..3).for_each(|idx| {
        circuit_evals[1 + idx] = circuit_evals[0] + gamma_square * final_ts_for_rows_evals[idx];
        circuit_evals[5 + idx] = circuit_evals[4] + gamma_square * final_ts_for_cols_evals[idx];
    });

    let mut final_claimed_values = F::ZERO;
    for c in 0..n_circuits {
        final_claimed_values += combiners[c] * circuit_evals[c]
    }
    assert_eq!(
        expected_eval, final_claimed_values,
        "input layer check failed of first circuit"
    )
}

pub fn input_layer_check2<F: PrimeField + Serialize + DeserializeOwned>(
    gamma_tau: &Vec<F>,
    expected_eval: F,
    combiners: &Vec<F>,
    n_circuits: usize,
    evaluations: &Vec<&F>,
) {
    let gamma_square = gamma_tau[0].square();
    let mut circuit_evals = vec![F::ZERO; n_circuits];
    (0..3).for_each(|idx| {
        circuit_evals[idx + 3] = *evaluations[idx]
            + gamma_tau[0] * *evaluations[3 + idx]
            + gamma_square * *evaluations[6 + idx]
            - gamma_tau[1];
        circuit_evals[idx] = circuit_evals[idx + 3] + gamma_square;
        circuit_evals[idx + 9] = *evaluations[9 + idx]
            + gamma_tau[0] * *evaluations[12 + idx]
            + gamma_square * *evaluations[15 + idx]
            - gamma_tau[1];
        circuit_evals[idx + 6] = circuit_evals[idx + 9] + gamma_square;
    });

    let mut final_claimed_values = F::ZERO;
    for c in 0..n_circuits {
        final_claimed_values += combiners[c] * circuit_evals[c]
    }
    assert_eq!(
        expected_eval, final_claimed_values,
        "Input layer check failed of second circuit"
    )
}
