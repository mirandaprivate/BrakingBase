use std::time::Instant;

use super::batcheval::batch_eval_proof;
use super::helper::{eR1CSmetadata, sparse_matrix_multiply, SparseRep};
use crate::pcs::multilinear::brakingbase::{Brakingbase, BrakingbaseProverParams, BrakingbaseSpec};
use crate::pcs::multilinear::brakingbase_helper::{par_fold_by_msb, point_to_tensor};
use crate::pcs::PolynomialCommitmentScheme;
use crate::poly::Polynomial;
use crate::util::hash::Hash;
use crate::{poly::multilinear::MultilinearPolynomial, util::transcript::TranscriptWrite};
use ff::{Field, PrimeField};
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
        rx_basis_evals,
        transcript,
    );

    let rx = fsrp.clone();
    let ry = par_srp.clone();

    // let E_eval_proof = evaluate(E.as_coeffs(), &rx, srs);
    // let W_eval_proof = evaluate(W.as_coeffs(), &ry, srs);

    let matrix_eval_point = [rx, ry].concat();

    let metadatas = vec![metadatas.A, metadatas.B, metadatas.C];

    let Eval_Proofs = batch_eval_proof::<F, H, S>(metadatas, &matrix_eval_point, pp, transcript);
    // eR1CStranscript::new(
    //     first_sum_check_transcript,
    //     par_sum_check_transcript,
    //     Eval_Proofs,
    //     E_eval_proof,
    //     W_eval_proof,
    // )
}

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
    // let mut E = MultPolynomial::new(E.as_coeffs().clone().to_vec());

    let mut Az = sparse_matrix_multiply(A, &z);

    let mut Bz = sparse_matrix_multiply(B, &z);

    let mut Cz = sparse_matrix_multiply(C, &z);

    let sum_check_rounds = z.len().trailing_zeros() as usize;

    // let tau = transcript.get_random_points(sum_check_rounds);
    let tau = transcript.squeeze_challenges(sum_check_rounds);

    // let mut fourcoeffs = MultPolynomial::new(compute_coeff(&tau));
    let mut fourcoeffs = point_to_tensor(1, &tau).1;
    let mut random_points = vec![F::ZERO; sum_check_rounds];
    let f_2_inv = F::from(2 as u64).invert().unwrap();
    let f_3_inv = F::from(3 as u64).invert().unwrap();
    let f_6_inv = F::from(6 as u64).invert().unwrap();
    for round in 0..sum_check_rounds {
        let mut eval = [F::ZERO; 4];
        let halfsize = 1 << (sum_check_rounds - round - 1);
        for k in 0..halfsize {
            let k_halfsize = k + halfsize;
            //eval at 0
            eval[0] += fourcoeffs[k] * (Az[k] * Bz[k] - (*u * Cz[k] + E[k]));

            //eval at 1
            eval[1] += fourcoeffs[k_halfsize]
                * (Az[k_halfsize] * Bz[k_halfsize] - (*u * Cz[k_halfsize] + E[k_halfsize]));

            //eval at -1
            eval[2] += (fourcoeffs[k].double() - fourcoeffs[k_halfsize])
                * ((Az[k].double() - Az[k_halfsize]) * (Bz[k].double() - Bz[k_halfsize])
                    - (*u * (Cz[k].double() - Cz[k_halfsize]) + (E[k].double() - E[k_halfsize])));

            //eval at 2
            eval[3] += (fourcoeffs[k_halfsize].double() - fourcoeffs[k])
                * ((Az[k_halfsize].double() - Az[k]) * (Bz[k_halfsize].double() - Bz[k])
                    - (*u * (Cz[k_halfsize].double() - Cz[k]) + (E[k_halfsize].double() - E[k])));
        }
        //TODO:- Verify
        let a_1_f2_inv = eval[1] * f_2_inv;
        let a_0_f2_inv = eval[0] * f_2_inv;
        let a_2_f_6_inv = eval[3] * f_6_inv;
        let polynomial_current_round = [
            a_0_f2_inv - a_1_f2_inv + a_2_f_6_inv - eval[2] * f_6_inv,
            -eval[0] + a_1_f2_inv + eval[2] * f_2_inv,
            -a_0_f2_inv + eval[1] - a_2_f_6_inv - eval[2] * f_3_inv,
            eval[0],
        ]
        .to_vec();
        transcript
            .write_field_elements(&polynomial_current_round)
            .unwrap();

        // channel.reseed_with_Fs(&eval);

        // polynomials.push(Polynomial::new(eval.to_vec()));

        // let r_i = channel.get_random_point();
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
    // let random_coeffs = channel.get_random_points(3);
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

        // for p in 0..3 {
        //     interpolate(&mut eval[p])
        // }
        for p in 0..3 {
            for c in 0..3 {
                comb_poly[c] += random_coeffs[p] * eval[p][c];
            }
        }

        // par_sum_check_polys.push(Polynomial::new(comb_poly.clone()));

        // channel.reseed_with_Fs(&comb_poly);
        transcript.write_field_elements(&comb_poly).unwrap();

        let r_i = transcript.squeeze_challenge();
        // let r_i = channel.get_random_point();

        par_sum_check_random_points.push(r_i);

        Z = par_fold_by_msb(&Z, r_i);
        for p in 0..3 {
            batch[p] = par_fold_by_msb(&batch[p], r_i);
        }
    }
    transcript
        .write_field_elements(&[batch[0][0], batch[1][0], batch[2][0]])
        .unwrap();

    // ParSumCheckTranscript::new(
    //     par_sum_check_polys,
    //     par_sum_check_random_points,
    //     batch[0].get_coeff(0),
    //     batch[1].get_coeff(0),
    //     batch[2].get_coeff(0),
    //     Z.get_coeff(0),
    // )
    par_sum_check_random_points
}
// #[allow(non_snake_case)]
// pub fn verify_sat<F: PrimeField>(
//     transcript: eR1CStranscript,
//     commitments: eR1CSCommitments,
//     u: F,
//     PI: MultPolynomial,
//     pi_indices: Vec<usize>,
//     ver_key: &VerificationKey<BlsCurve>,
//     channel: &mut Channel,
// ) {
//     let rx = transcript.first_sum_check_transcript.random_points.clone();
//     let ry = transcript.par_sum_check_transcript.random_points.clone();

//     let Az_claimed_val = transcript.first_sum_check_transcript.Az_claimed_val;
//     let Bz_claimed_val = transcript.first_sum_check_transcript.Bz_claimed_val;
//     let Cz_claimed_val = transcript.first_sum_check_transcript.Cz_claimed_val;
//     let E_final_eval = transcript.E_eval_proof.evaluation;

//     let first_sum_check_final_eval =
//         (Az_claimed_val * Bz_claimed_val) - (u * Cz_claimed_val + E_final_eval);

//     initial_sum_check_verification(
//         &transcript.first_sum_check_transcript,
//         first_sum_check_final_eval,
//         channel,
//     );

//     let random_coeffs = channel.get_random_points(3);

//     let eval = random_coeffs[0] * Az_claimed_val
//         + random_coeffs[1] * Bz_claimed_val
//         + random_coeffs[2] * Cz_claimed_val;

//     let PI_eval = evaluate_PI(pi_indices, PI, &ry, ry.len());
//     let eval_point = [rx, ry.clone()].concat();
//     let W_eval = transcript.W_eval_proof.evaluation;

//     let Z_final_eval = PI_eval + W_eval;

//     let A_claimed_val = transcript.par_sum_check_transcript.A_claimed_val;
//     let B_claimed_val = transcript.par_sum_check_transcript.B_claimed_val;
//     let C_claimed_val = transcript.par_sum_check_transcript.C_claimed_val;

//     let par_sum_check_final_eval = Z_final_eval
//         * (random_coeffs[0] * A_claimed_val
//             + random_coeffs[1] * B_claimed_val
//             + random_coeffs[2] * C_claimed_val);

//     par_sum_check_verification(
//         transcript.par_sum_check_transcript,
//         eval,
//         par_sum_check_final_eval,
//         channel,
//     );

//     // Verifying the claimed values.
//     let (
//         sum_check_batch_commit,
//         _sum_check_batch_eval,
//         sum_check_eval_proof,
//         sum_check_random_points,
//         gkr_commit1,
//         gkr_commit2,
//         gkr_eval1,
//         _gkr_eval2,
//         gkr_batch_eval_proof1,
//         gkr_batch_eval_proof2,
//         gkr_final_layer_point1,
//         gkr_final_layer_point2,
//     ) = batch_opening(
//         vec![A_claimed_val, B_claimed_val, C_claimed_val],
//         transcript.BatchProof,
//         vec![commitments.A, commitments.B, commitments.C],
//         &eval_point,
//         channel,
//     );

//     let mut commits = Vec::new();
//     let mut points = Vec::new();
//     let mut proofs = Vec::new();
//     let mut evals = Vec::new();

//     commits.push(commitments.E);
//     points.push(transcript.first_sum_check_transcript.random_points);
//     proofs.push(transcript.E_eval_proof);
//     evals.push(E_final_eval);

//     commits.push(commitments.W);
//     points.push(ry);
//     proofs.push(transcript.W_eval_proof);
//     evals.push(W_eval);
//     commits.push(gkr_commit1);
//     points.push(gkr_final_layer_point1);
//     proofs.push(gkr_batch_eval_proof1);
//     evals.push(gkr_eval1);

//     let Fs = channel.get_random_points(4);
//     batch_verify_var_openings(&commits, &evals, &proofs, &points, &Fs, ver_key);

//     verify(
//         &sum_check_batch_commit,
//         &sum_check_eval_proof,
//         &sum_check_random_points,
//         ver_key,
//     );

//     verify(
//         &gkr_commit2,
//         &gkr_batch_eval_proof2,
//         &gkr_final_layer_point2,
//         ver_key,
//     );
// }

// pub fn initial_sum_check_verification(
//     transcript: &InitialSumCheckTranscript,
//     final_evaluation: F,
//     channel: &mut Channel,
// ) {
//     let polynomials = &transcript.polynomials;
//     let rounds = polynomials.len();
//     let tau = channel.get_random_points(rounds);

//     let mut current_sum = F::ZERO;

//     let mut r = vec![F::ZERO; rounds];
//     for i in 0..rounds {
//         let poly = polynomials[i].get_coefficients();
//         assert_eq!(
//             current_sum,
//             eval(poly, F::ZERO) + eval(poly, F::ONE),
//             "f(0) + f(1) did not match binding at round {:?} in the eR1CS initial sum check",
//             i
//         );

//         channel.reseed_with_Fs(poly);
//         let r_i = channel.get_random_point();
//         r[i] = r_i;
//         current_sum = eval(poly, r_i)
//     }
//     let eq = evaluate_eq(tau, r);
//     assert_eq!(
//         current_sum,
//         eq * final_evaluation,
//         "Final assertion in eR1CS initial sum check failed"
//     )
// }

// pub fn par_sum_check_verification(
//     transcript: ParSumCheckTranscript,
//     initial_evaluation: F,
//     final_evaluation: F,
//     channel: &mut Channel,
// ) {
//     let polynomials = &transcript.polynomials;
//     let rounds = polynomials.len();

//     let mut current_sum = initial_evaluation;

//     let mut r = vec![F::ZERO; rounds];
//     for i in 0..rounds {
//         let poly = polynomials[i].get_coefficients();
//         assert_eq!(
//             current_sum,
//             eval(poly, F::ZERO) + eval(poly, F::ONE),
//             "f(0) + f(1) did not match binding at round {:?} in the second eR1CS sum check",
//             i
//         );

//         channel.reseed_with_Fs(poly);
//         let r_i = channel.get_random_point();
//         r[i] = r_i;
//         current_sum = eval(poly, r_i)
//     }

//     assert_eq!(
//         current_sum, final_evaluation,
//         "Final assertion in eR1CS second sum check failed"
//     )
// }
// #[allow(non_snake_case)]
// pub fn evaluate_PI(pi_indices: Vec<usize>, PI: MultPolynomial, point: &Vec<F>, bits: usize) -> F {
//     let mut eval = F::ZERO;

//     for j in 0..PI.len() {
//         let mut basis_eval = PI.get_coeff(j);
//         for i in (0..bits).rev() {
//             if (pi_indices[j] >> (i)) & 1 == 1 {
//                 basis_eval *= point[bits - i - 1]
//             } else {
//                 basis_eval *= F::ONE - point[bits - i - 1]
//             }
//         }
//         eval += basis_eval;
//     }
//     eval
// }

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

#[test]
fn test_basis() {
    use crate::pcs::multilinear::brakingbase_helper::eq;
    use halo2_curves::bn256::Fr;
    use rand::rngs::OsRng;
    let size = 10;
    let mut rng = OsRng;
    let random_points: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();
    let start_time = Instant::now();
    let eq_bases = point_to_tensor::<Fr>(1, &random_points).1;
    println!("time 1 {:?}", start_time.elapsed());
    let mut eq_bases2 = Vec::new();
    let start_time = Instant::now();
    for i in 0..(1 << size) {
        eq_bases2.push(eq(i, &random_points));
    }
    println!("time 2 {:?}", start_time.elapsed());

    assert_eq!(eq_bases, eq_bases2);
}
