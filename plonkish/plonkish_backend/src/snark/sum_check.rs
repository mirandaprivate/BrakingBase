use super::helper::{sparse_matrix_multiply, SparseRep};
use crate::pcs::multilinear::brakingbase::{Brakingbase, BrakingbaseSpec};
use crate::pcs::multilinear::brakingbase_helper::{
    eval, evaluate_eq, par_fold_by_msb, point_to_tensor,
};
use crate::pcs::PolynomialCommitmentScheme;
use crate::poly::multilinear::MultilinearPolynomial;
use crate::poly::Polynomial;
use crate::util::hash::Hash;
use crate::util::transcript::{TranscriptRead, TranscriptWrite};
use ff::PrimeField;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{de::DeserializeOwned, Serialize};
#[allow(non_snake_case)]
pub fn first_layer_sum_check<F, H, S>(
    A: &SparseRep<F>,
    B: &SparseRep<F>,
    C: &SparseRep<F>,
    u: &F,
    z: &Vec<F>,
    mut E: Vec<F>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> Vec<F>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let mut Az = sparse_matrix_multiply(A, &z);

    let mut Bz = sparse_matrix_multiply(B, &z);

    let mut Cz = sparse_matrix_multiply(C, &z);

    let sum_check_rounds = z.len().trailing_zeros() as usize;

    let tau = transcript.squeeze_challenges(sum_check_rounds);

    let mut fourcoeffs = point_to_tensor(1, &tau).1;
    let mut random_points = vec![F::ZERO; sum_check_rounds];

    let f_2_inv = F::from(2 as u64).invert().unwrap();
    let f_3_inv = F::from(3 as u64).invert().unwrap();
    let f_6_inv = F::from(6 as u64).invert().unwrap();

    for round in 0..sum_check_rounds {
        let halfsize = fourcoeffs.len() / 2;
        let (a_0, a_1, a_2, a_minus_one) = (0..halfsize)
            .into_par_iter()
            .map(|k| {
                let k_halfsize = k + halfsize;
                let a_0 = fourcoeffs[k] * (Az[k] * Bz[k] - (*u * Cz[k] + E[k]));

                let a_1 = fourcoeffs[k_halfsize]
                    * (Az[k_halfsize] * Bz[k_halfsize] - (*u * Cz[k_halfsize] + E[k_halfsize]));

                let a_2 = (fourcoeffs[k_halfsize].double() - fourcoeffs[k])
                    * ((Az[k_halfsize].double() - Az[k]) * (Bz[k_halfsize].double() - Bz[k])
                        - (*u * (Cz[k_halfsize].double() - Cz[k])
                            + (E[k_halfsize].double() - E[k])));

                let a_minus_one = (fourcoeffs[k].double() - fourcoeffs[k_halfsize])
                    * ((Az[k].double() - Az[k_halfsize]) * (Bz[k].double() - Bz[k_halfsize])
                        - (*u * (Cz[k].double() - Cz[k_halfsize])
                            + (E[k].double() - E[k_halfsize])));

                (a_0, a_1, a_2, a_minus_one)
            })
            .reduce_with(|(acc0, acc1, acc2, acc3), (a_0, a_1, a_2, a_minus_one)| {
                (acc0 + a_0, acc1 + a_1, acc2 + a_2, acc3 + a_minus_one)
            })
            .unwrap();

        let a_1_f2_inv = a_1 * f_2_inv;
        let a_0_f2_inv = a_0 * f_2_inv;
        let a_2_f_6_inv = a_2 * f_6_inv;

        let polynomial_current_round = [
            a_0_f2_inv - a_1_f2_inv + a_2_f_6_inv - a_minus_one * f_6_inv,
            -a_0 + a_1_f2_inv + a_minus_one * f_2_inv,
            -a_0_f2_inv + a_1 - a_2_f_6_inv - a_minus_one * f_3_inv,
            a_0,
        ]
        .to_vec();

        transcript
            .write_field_elements(&polynomial_current_round)
            .unwrap();

        let r_i = transcript.squeeze_challenge();
        random_points[round] = r_i;

        fourcoeffs = par_fold_by_msb(&fourcoeffs, r_i);
        Az = par_fold_by_msb(&Az, r_i);
        Bz = par_fold_by_msb(&Bz, r_i);
        Cz = par_fold_by_msb(&Cz, r_i);
        E = par_fold_by_msb(&E, r_i);
    }

    transcript
        .write_field_elements(&[Az[0], Bz[0], Cz[0], E[0]])
        .unwrap();

    random_points
}

#[allow(non_snake_case)]
pub fn parallel_sum_checks<F, H, S>(
    A: &SparseRep<F>,
    B: &SparseRep<F>,
    C: &SparseRep<F>,
    mut Z: Vec<F>,
    rx_basis_evals: Vec<F>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> Vec<F>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let z_len = Z.len();
    let sum_check_rounds = z_len.trailing_zeros() as usize;

    let A_rx: Vec<F> = A.bind_row_variable(&rx_basis_evals, z_len);

    let B_rx: Vec<F> = B.bind_row_variable(&rx_basis_evals, z_len);

    let C_rx: Vec<F> = C.bind_row_variable(&rx_basis_evals, z_len);

    let mut batch = vec![A_rx, B_rx, C_rx];
    let random_coeffs = transcript.squeeze_challenges(3);

    let f_2_inv = F::from(2 as u64).invert().unwrap();

    let mut par_sum_check_random_points = Vec::new();
    for round in 0..sum_check_rounds {
        let mut eval = vec![vec![F::ZERO; 3]; 3];
        let halfsize = 1 << (sum_check_rounds - round - 1);

        let mut comb_poly = vec![F::ZERO; 3];
        for k in 0..halfsize {
            let k_halfsize = k + halfsize;
            let temp = Z[k_halfsize].double() - Z[k];
            for p in 0..3 {
                eval[p][0] += batch[p][k] * Z[k];
                eval[p][1] += batch[p][k_halfsize] * Z[k_halfsize];

                eval[p][2] += (batch[p][k_halfsize].double() - batch[p][k]) * temp;
            }
        }
        for p in 0..3 {
            let a_0_f_2_inv = eval[p][0] * f_2_inv;
            let a_2_f_2_inv = eval[p][2] * f_2_inv;
            eval[p] = [
                a_0_f_2_inv - eval[p][1] + a_2_f_2_inv,
                -(a_0_f_2_inv.double() + a_0_f_2_inv) + eval[p][1].double() - a_2_f_2_inv,
                eval[p][0],
            ]
            .to_vec();
        }

        for p in 0..3 {
            for c in 0..3 {
                comb_poly[c] += random_coeffs[p] * eval[p][c];
            }
        }

        transcript.write_field_elements(&comb_poly).unwrap();

        let r_i = transcript.squeeze_challenge();

        par_sum_check_random_points.push(r_i);

        Z = par_fold_by_msb(&Z, r_i);
        for p in 0..3 {
            batch[p] = par_fold_by_msb(&batch[p], r_i);
        }
    }
    transcript
        .write_field_elements(&[batch[0][0], batch[1][0], batch[2][0]])
        .unwrap();

    par_sum_check_random_points
}

pub fn initial_sum_check_verification<F, H, S>(
    num_const: usize,
    u: F,
    transcript: &mut impl TranscriptRead<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> (Vec<F>, Vec<F>)
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let rounds = num_const.trailing_zeros() as usize;
    let tau = transcript.squeeze_challenges(rounds);

    let mut current_sum = F::ZERO;
    let mut r = vec![F::ZERO; rounds];
    for i in 0..rounds {
        let poly = transcript.read_field_elements(4).unwrap();
        assert_eq!(
            current_sum,
            poly[3].double() + poly[2] + poly[1] + poly[0],
            "f(0) + f(1) did not match binding at round {:?} in the eR1CS initial sum check",
            i
        );
        r[i] = transcript.squeeze_challenge();

        current_sum = poly[3] + (poly[2] + (poly[1] + poly[0] * r[i]) * r[i]) * r[i];
    }

    let eq = evaluate_eq(&tau, &r);
    let eval = transcript.read_field_elements(4).unwrap();
    let final_evaluation = (eval[0] * eval[1]) - (u * eval[2] + eval[3]);
    assert_eq!(
        current_sum,
        eq * final_evaluation,
        "Final assertion in eR1CS initial sum check failed"
    );
    (eval, r)
}

pub fn par_sum_check_verification<F, H, S>(
    num_const: usize,
    pi_indices: Vec<usize>,
    PI: MultilinearPolynomial<F>,
    random_coeffs: &Vec<F>,
    initial_evaluation: F,
    transcript: &mut impl TranscriptRead<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> (Vec<F>, Vec<F>, F)
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let mut current_sum = initial_evaluation;

    let rounds = num_const.trailing_zeros() as usize;
    let mut r = vec![F::ZERO; rounds];

    for i in 0..rounds {
        let poly = transcript.read_field_elements(3).unwrap();
        assert_eq!(
            current_sum,
            poly[2].double() + poly[1] + poly[0],
            "f(0) + f(1) did not match binding at round {:?} in the second eR1CS sum check",
            i
        );
        r[i] = transcript.squeeze_challenge();
        current_sum = poly[2] + (poly[1] + poly[0] * r[i]) * r[i];
    }
    let a_b_c_evals = transcript.read_field_elements(3).unwrap();
    let PI_eval = evaluate_PI(pi_indices, PI, &r);

    let w_eval = transcript.read_field_element().unwrap();
    let Z_final_eval = PI_eval + w_eval;

    let final_evaluation = Z_final_eval
        * (random_coeffs[0] * a_b_c_evals[0]
            + random_coeffs[1] * a_b_c_evals[1]
            + random_coeffs[2] * a_b_c_evals[2]);

    assert_eq!(
        current_sum, final_evaluation,
        "Final assertion in eR1CS second sum check failed"
    );
    (r, a_b_c_evals, w_eval)
}

pub fn matrix_eval_sum_check<F, H, S>(
    mut val: Vec<Vec<F>>,
    mut e_rx: Vec<Vec<F>>,
    mut e_ry: Vec<Vec<F>>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> Vec<F>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let sum_check_rounds = val[0].len().trailing_zeros() as usize;
    let random_coeffs = transcript.squeeze_challenges(3);
    let mut sum_check_random_points: Vec<F> = vec![F::ZERO; sum_check_rounds];

    let f_6 = F::from(6 as u64);
    let f_2_inv = F::from(2 as u64).invert().unwrap();
    let f_3_inv = F::from(3 as u64).invert().unwrap();
    let f_6_inv = f_6.invert().unwrap();

    for i in 0..sum_check_rounds {
        let mut eval = vec![[F::ZERO; 4]; 3];
        for c in 0..val.len() {
            let halfsize = val[c].len() / 2;
            let (a_0, a_1, a_2, a_minus_one) = (0..halfsize)
                .into_par_iter()
                .map(|iter| {
                    let iter2 = iter + halfsize;

                    let a_0 = val[c][iter] * e_rx[c][iter] * e_ry[c][iter];

                    let a_1 = val[c][iter2] * e_rx[c][iter2] * e_ry[c][iter2];

                    let a_2 = (val[c][iter2].double() - val[c][iter])
                        * (e_rx[c][iter2].double() - e_rx[c][iter])
                        * (e_ry[c][iter2].double() - e_ry[c][iter]);

                    let a_minus_one = (val[c][iter].double() - val[c][iter2])
                        * (e_rx[c][iter].double() - e_rx[c][iter2])
                        * (e_ry[c][iter].double() - e_ry[c][iter2]);

                    (a_0, a_1, a_2, a_minus_one)
                })
                .reduce_with(|(acc0, acc1, acc2, acc3), (a_0, a_1, a_2, a_minus_one)| {
                    (acc0 + a_0, acc1 + a_1, acc2 + a_2, acc3 + a_minus_one)
                })
                .unwrap();
            let a_1_f2_inv = a_1 * f_2_inv;
            let a_0_f2_inv = a_0 * f_2_inv;
            let a_2_f_6_inv = a_2 * f_6_inv;
            eval[c] = [
                a_0_f2_inv - a_1_f2_inv + a_2_f_6_inv - a_minus_one * f_6_inv,
                -a_0 + a_1_f2_inv + a_minus_one * f_2_inv,
                -a_0_f2_inv + a_1 - a_2_f_6_inv - a_minus_one * f_3_inv,
                a_0,
            ];
        }

        let mut combined_poly = [F::ZERO; 4].to_vec();
        for k in 0..4 {
            for c in 0..val.len() {
                combined_poly[k] += random_coeffs[c] * eval[c][k]
            }
        }
        transcript.write_field_elements(&combined_poly).unwrap();

        sum_check_random_points[i] = transcript.squeeze_challenge();

        for k in 0..val.len() {
            e_rx[k] = par_fold_by_msb(&e_rx[k], sum_check_random_points[i]);
            e_ry[k] = par_fold_by_msb(&e_ry[k], sum_check_random_points[i]);
            val[k] = par_fold_by_msb(&val[k], sum_check_random_points[i]);
        }
    }

    transcript
        .write_field_elements(&[e_rx[0][0], e_rx[1][0], e_rx[2][0]])
        .unwrap();
    transcript
        .write_field_elements(&[e_ry[0][0], e_ry[1][0], e_ry[2][0]])
        .unwrap();
    transcript
        .write_field_elements(&[val[0][0], val[1][0], val[2][0]])
        .unwrap();

    sum_check_random_points
}

pub fn matrix_eval_sum_check_verifier<F, H, S>(
    num_const: usize,
    sparsity: usize,
    initial_evaluation: F,
    random_coeffs: Vec<F>,
    transcript: &mut impl TranscriptRead<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>)
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let rounds = ((num_const * sparsity) as u32).trailing_zeros() as usize;
    let mut current_sum = initial_evaluation;

    let mut r = vec![F::ZERO; rounds];
    for i in 0..rounds {
        let poly = transcript.read_field_elements(4).unwrap();
        assert_eq!(
            current_sum,
            poly[3].double() + poly[2] + poly[1] + poly[0],
            "f(0) + f(1) did not match binding at round {:?}",
            i
        );

        r[i] = transcript.squeeze_challenge();
        current_sum = poly[3] + (poly[2] + (poly[1] + poly[0] * r[i]) * r[i]) * r[i];
    }

    let mut final_eval = F::ZERO;
    let e_rx_evals = transcript.read_field_elements(3).unwrap();
    let e_ry_evals = transcript.read_field_elements(3).unwrap();
    let val_evals = transcript.read_field_elements(3).unwrap();

    for i in 0..3 {
        final_eval += random_coeffs[i] * (e_rx_evals[i] * e_ry_evals[i] * val_evals[i])
    }

    assert_eq!(
        current_sum, final_eval,
        "Final sum check verification failed in Spartan"
    );
    (r, e_rx_evals, e_ry_evals, val_evals)
}

#[allow(non_snake_case)]
pub fn evaluate_PI<F: PrimeField + Serialize + DeserializeOwned>(
    pi_indices: Vec<usize>,
    PI: MultilinearPolynomial<F>,
    point: &Vec<F>,
) -> F {
    let bits = point.len();
    let mut eval = F::ZERO;

    for j in 0..PI.evals().len() {
        let mut basis_eval = PI.evals()[j];
        for i in (0..bits).rev() {
            if (pi_indices[j] >> (i)) & 1 == 1 {
                basis_eval *= point[bits - i - 1]
            } else {
                basis_eval *= F::ONE - point[bits - i - 1]
            }
        }
        eval += basis_eval;
    }
    eval
}

pub fn batch_sum_check_verifier<F, H, S>(
    mut r_x: Vec<Vec<F>>,
    claimed_eval: F,
    transcript: &mut impl TranscriptRead<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
    batch_sc_rc: &Vec<F>,
) where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let mut actual_result = claimed_eval;
    let mut r_y = Vec::new();

    for var in 0..r_x[2].len() {
        let poly = transcript.read_field_elements(3).unwrap();
        let mut previous_result = poly[0];
        for i in 0..poly.len() {
            previous_result += poly[i];
        }
        assert_eq!(actual_result, previous_result, "failed at round {}", var);

        let r = transcript.squeeze_challenge();
        r_y.push(r);

        actual_result = eval(&poly, r);
    }
    let rows_evals = transcript.read_field_elements(3).unwrap();
    let cols_evals = transcript.read_field_elements(3).unwrap();
    let read_ts_for_rows_evals = transcript.read_field_elements(3).unwrap();
    let read_ts_for_cols_evals = transcript.read_field_elements(3).unwrap();
    let e_rx_evals = transcript.read_field_elements(3).unwrap();
    let e_ry_evals = transcript.read_field_elements(3).unwrap();
    let final_ts_for_rows_evals = transcript.read_field_elements(3).unwrap();
    let final_ts_for_cols_evals = transcript.read_field_elements(3).unwrap();
    let val_evals = transcript.read_field_elements(3).unwrap();
    let E_eval = transcript.read_field_element().unwrap();
    let W_eval = transcript.read_field_element().unwrap();

    extend_if_required::<F>(&mut r_x);

    let r_x_evals = r_x
        .iter()
        .map(|rx| evaluate_eq(rx, &r_y))
        .collect::<Vec<F>>();

    let mut final_claim =
        batch_sc_rc[0] * E_eval * r_x_evals[0] + batch_sc_rc[1] * W_eval * r_x_evals[1];

    let mut temp = F::ZERO;
    batch_sc_rc
        .iter()
        .skip(2)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * e_rx_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(5)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * e_ry_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(8)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * val_evals[idx];
        });
    final_claim += temp * r_x_evals[2];

    let mut temp = F::ZERO;
    batch_sc_rc
        .iter()
        .skip(11)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * e_rx_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(14)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * e_ry_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(17)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * rows_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(20)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * cols_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(23)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * read_ts_for_rows_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(26)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * read_ts_for_cols_evals[idx];
        });
    final_claim += temp * r_x_evals[3];

    let mut temp = F::ZERO;
    batch_sc_rc
        .iter()
        .skip(29)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * final_ts_for_rows_evals[idx];
        });
    batch_sc_rc
        .iter()
        .skip(32)
        .take(3)
        .enumerate()
        .for_each(|(idx, coeff)| {
            temp += *coeff * final_ts_for_cols_evals[idx];
        });
    final_claim += temp * r_x_evals[4];

    assert_eq!(
        actual_result, final_claim,
        "Final assertion failed in batch sum check"
    );
}

fn extend_if_required<F: PrimeField + Serialize + DeserializeOwned>(rx: &mut Vec<Vec<F>>) {
    let max_size = rx[2].len();
    let diff = max_size - rx[0].len();
    if diff != 0 {
        let temp1 = rx[0].clone();
        let temp2 = rx[1].clone();
        let temp3 = rx[4].clone();
        rx[0].reverse();
        rx[1].reverse();
        rx[4].reverse();

        rx[0].extend(temp1[0..diff].to_vec());
        rx[1].extend(temp2[0..diff].to_vec());
        rx[4].extend(temp3[0..diff].to_vec());

        rx[0].reverse();
        rx[1].reverse();
        rx[4].reverse();
    }
}
