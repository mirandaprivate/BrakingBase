use super::helper::{compute_fourier_bases, len_4_interpolate};
use crate::pcs::multilinear::brakingbase_helper::{eval, evaluate_eq, par_fold_by_msb};
use crate::util::hash::Hash;
use crate::util::transcript::FieldTranscriptRead;
use crate::{
    pcs::{
        multilinear::brakingbase::{Brakingbase, BrakingbaseSpec},
        PolynomialCommitmentScheme,
    },
    util::transcript::TranscriptWrite,
};
use ff::PrimeField;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;
use serde::de::DeserializeOwned;
use serde::Serialize;

#[derive(Clone)]
pub struct GkrTranscript<F: PrimeField> {
    pub final_evaluations: Vec<Vec<F>>,
    pub claimed_values: Vec<Vec<Vec<F>>>,
    pub polynomials: Vec<Vec<Vec<F>>>,
}

impl<F: PrimeField> GkrTranscript<F> {
    pub fn new(
        final_evaluations: Vec<Vec<F>>,
        claimed_values: Vec<Vec<Vec<F>>>,
        polynomials: Vec<Vec<Vec<F>>>,
    ) -> GkrTranscript<F> {
        GkrTranscript {
            final_evaluations,
            claimed_values,
            polynomials,
        }
    }
}
//Prover for the sub-circuit corresponding to the leaf layer inputs of length of the table.
pub fn gkr_prover<F: PrimeField + Serialize + DeserializeOwned, H: Hash, S: BrakingbaseSpec>(
    circuits: &Vec<&Vec<Vec<F>>>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> Vec<F> {
    let depth = circuits[0].len() - 1;
    circuits.iter().for_each(|circuit| {
        assert_eq!(depth, circuit.len() - 1, "Circuits do not have same depth")
    });

    let n_circuits = circuits.len();
    println!("n_circuits is {:?}", n_circuits);
    //This vector contains the values of the circuits at depth 1 i.e. the layer below the output layer.
    let mut final_values = Vec::new();
    for c in 0..n_circuits {
        let circuit_layer_1 = &circuits[c][depth - 1];
        final_values.extend(vec![circuit_layer_1[0], circuit_layer_1[1]]);
    }
    transcript.write_field_elements(&final_values).unwrap();

    let mut initial_random_point = vec![transcript.squeeze_challenge()];

    //We are verifying the circuit evaluations in a batched manner by taking a linear combination of
    //the gate evaluation MLEs for each circuit, at each layer.
    let random_coeff = transcript.squeeze_challenges(n_circuits);

    for layer in (1..depth).rev() {
        //The dense representation of the lagrange basis functions evaluated at the random point.
        let mut lagrange_bases_eval = compute_fourier_bases(initial_random_point.clone());

        let current_depth = depth - layer;
        let layer_size = 1 << current_depth;

        //Random points generated over the sum check instance.
        let mut sum_check_random_points: Vec<F> = vec![F::ONE; current_depth + 1];

        //Dense representation of the next depth's left child gate and right child gate mle's respectively W_{d-1}(x;0), W_{d-1}(x:1)
        let mut child_left_extension = vec![vec![F::ONE; layer_size]; n_circuits];
        let mut child_right_extension = vec![vec![F::ONE; layer_size]; n_circuits];

        child_left_extension
            .par_iter_mut()
            .zip(child_right_extension.par_iter_mut())
            .enumerate()
            .for_each(|(c, (left, right))| {
                for i in 0..layer_size {
                    left[i] = circuits[c][layer - 1][2 * i];
                    right[i] = circuits[c][layer - 1][2 * i + 1]
                }
            });

        //This is the sum check instance for this layer.
        for i in 0..current_depth {
            //This contains the circuit-wise values for univariate polynomial to be sent evaluated at 0,1,-1 and 2.
            let mut eval = vec![[F::ZERO; 4]; n_circuits];

            //This contains the coefficients of the linear combination of the circuit-wise evaluations.
            let mut combined_polynomial = vec![F::ZERO; 4];
            let halfsize = layer_size >> (i + 1);

            //We evaluate the dense representation of the multilinear polynomial that is the linear coefficient of lagrange_bases_eval
            //when viewed as a polynomial in one variable.
            let mut lagrange_bases_lin_coeff = vec![F::ZERO; halfsize];

            lagrange_bases_lin_coeff
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, fc_coeff)| {
                    *fc_coeff = lagrange_bases_eval[j + halfsize] - lagrange_bases_eval[j]
                });

            //We compute the sum over all binary strings for the remaining variables, parallelised over the circuits.
            eval.par_iter_mut().enumerate().for_each(|(c, eval_c)| {
                (eval_c[0], eval_c[1], eval_c[2], eval_c[3]) = (0..halfsize)
                    .into_par_iter()
                    .map(|j| {
                        //We use the fact that for any multilinear polynomial W in variables, x_1, ..., x_d+1,
                        //W(x_1, ..., x_d+1) = (1-x_1).W(x_1, ... , x_d,0) + x_1.W(x_1,...,x_d,1).
                        let child_left_temp =
                            child_left_extension[c][j + halfsize] - child_left_extension[c][j];
                        let child_right_temp =
                            child_right_extension[c][j + halfsize] - child_right_extension[c][j];

                        rayon::join(
                            || {
                                rayon::join(
                                    ||
                        //Evaluation at 0
                         lagrange_bases_eval[j]
                                * child_left_extension[c][j]
                                * child_right_extension[c][j],
                                    ||    //Evaluation at 1
                            lagrange_bases_eval[j + halfsize]
                                * child_left_extension[c][j + halfsize]
                                * child_right_extension[c][j + halfsize],
                                )
                            },
                            || {
                                rayon::join(
                                    ||  //Evaluation at -1
                            (lagrange_bases_eval[j] - lagrange_bases_lin_coeff[j])
                                * (child_left_extension[c][j] - child_left_temp)
                                * (child_right_extension[c][j] - child_right_temp),
                                    ||//Evaluation at 2
                            (lagrange_bases_eval[j + halfsize] + lagrange_bases_lin_coeff[j])
                                * (child_left_extension[c][j + halfsize] + child_left_temp)
                                * (child_right_extension[c][j + halfsize] + child_right_temp),
                                )
                            },
                        )
                    })
                    .fold_with(
                        (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
                        |(acc0, acc1, acc2, acc3), val| {
                            (
                                acc0 + val.0 .0,
                                acc1 + val.0 .1,
                                acc2 + val.1 .0,
                                acc3 + val.1 .1,
                            )
                        },
                    )
                    .reduce_with(|(acc0, acc1, acc2, acc3), (val0, val1, val2, val3)| {
                        (acc0 + val0, acc1 + val1, acc2 + val2, acc3 + val3)
                    })
                    .unwrap();
            });

            //We conduct the inverse linear transform fromthe evaluations to get the coefficients of the circuit-wise polynomials.
            eval.par_iter_mut()
                .for_each(|eval_c| len_4_interpolate(eval_c));

            //We add the linear combination of the coefficients to get the batched polynomial for the sum check verifieer to verify.
            combined_polynomial
                .par_iter_mut()
                .enumerate()
                .for_each(|(k, poly)| {
                    for c in 0..n_circuits {
                        *poly += random_coeff[c] * eval[c][k];
                    }
                });

            //The next round's random point is obtained by reseeding with this round's batched polynomial as a vector of scalars
            // and then drawing from the channel.
            transcript
                .write_field_elements(&combined_polynomial)
                .unwrap();

            let random_point = transcript.squeeze_challenge();
            sum_check_random_points[current_depth - i] = random_point;

            //Now we fix the leading variable of the the multilinear polynomials lagrange_bases_eval, child_left and child_right to get multilinear polynomials
            //in one less variable for the next sum check round
            //Since lagrange_bases_eval is a common computation accross circuits, we fold it in parallel separately.

            lagrange_bases_eval = par_fold_by_msb(&lagrange_bases_eval, random_point);

            //We fix the current leading variable in the sum check to the current random point of the protocol for the extension for the
            //child_left and child_right MLEs
            //
            child_left_extension
                .par_iter_mut()
                .zip(child_right_extension.par_iter_mut())
                .for_each(|(child_left_extension_c, child_right_extension_c)| {
                    *child_left_extension_c = par_fold_by_msb(child_left_extension_c, random_point);
                    *child_right_extension_c =
                        par_fold_by_msb(child_right_extension_c, random_point);
                });
        }

        //These are the variables that will contain the values for the appropriate circuit-wise linear-
        // combination for the claimed values, i.e. the scalar obtained after binding all but the last
        // variable to the random values obtained over the sum-check for the current layer. The last variable
        // is understood to be fixed to 0 and 1 respectively to obtain the MLEs of child_left and child_right
        // respectively.
        let mut mle_layer_evaluation = vec![F::ZERO; n_circuits * 2];
        mle_layer_evaluation
            .par_chunks_mut(2)
            .enumerate()
            .for_each(|(c, eval)| {
                eval[0] = child_left_extension[c][0];
                eval[1] = child_right_extension[c][0];
            });

        transcript
            .write_field_elements(&mle_layer_evaluation)
            .unwrap();

        //The line is of the form L(t) = (r_d_i;t), thus q(t)= W_{d-1}( L(t) ) is of degree 1 as W is linear in each variable.
        let r = transcript.squeeze_challenge();
        sum_check_random_points[0] = r;
        initial_random_point = sum_check_random_points
    }

    initial_random_point.reverse();
    initial_random_point
}

pub fn gkr_verifier<F: PrimeField + Serialize + DeserializeOwned>(
    depth: usize,
    transcript: &mut impl FieldTranscriptRead<F>,
    n_circuits: usize,
) -> (F, Vec<F>, Vec<F>, Vec<F>) {
    let final_evaluations = transcript.read_field_elements(n_circuits * 2).unwrap();
    let mut initial_random_point = vec![transcript.squeeze_challenge()];

    //Verifier obtains the random coefficients the prover uses.
    let random_coeff = transcript.squeeze_challenges(n_circuits);

    let mut binding_per_layer = F::ZERO;

    //The value for the claim of the first round of the protocol.
    for c in 0..n_circuits {
        binding_per_layer += random_coeff[c]
            * ((F::ONE - initial_random_point[0]) * final_evaluations[2 * c]
                + initial_random_point[0] * final_evaluations[2 * c + 1])
    }

    for d in 0..depth - 1 {
        let rounds = d + 1;
        let mut current_sum = binding_per_layer;
        let mut sum_check_random_points = vec![F::ONE; rounds + 1];

        for i in 0..rounds {
            let poly = transcript.read_field_elements(4).unwrap();
            assert_eq!(
                current_sum,
                poly[0].double() + poly[1] + poly[2] + poly[3],
                "Sum check failed on round {i} at depth {d}"
            );

            let r = transcript.squeeze_challenge();
            current_sum = eval::<F>(&poly, r);
            sum_check_random_points[rounds - i] = r;
        }

        let claimed_values = transcript.read_field_elements(n_circuits * 2).unwrap();
        let mut temp = F::ZERO;
        for c in 0..claimed_values.len() / 2 {
            temp += random_coeff[c] * (claimed_values[2 * c] * claimed_values[2 * c + 1])
        }

        let r = transcript.squeeze_challenge();
        sum_check_random_points[0] = r;

        let eq = evaluate_eq::<F>(
            &initial_random_point,
            &sum_check_random_points[1..].to_vec(),
        );
        assert_eq!(current_sum, eq * temp, "assertion failed at layer {d}");

        initial_random_point = sum_check_random_points;

        //After sum check for the layer completes successfully, the verifier can compute the challenge
        //for the next layer using the claimed values sent by the prover. Which correspond to,
        //W(x_1, x_2, ..., x_d, 0),W(x_1,x_2,...,x_d,1) respectively.
        //For any multilinear polynomial W in variables, x_1, ..., x_d+1.
        //W(x_1, ..., x_d+1) = (1-x_1).W(x_1, ... , x_d,0) + x_1.W(x_1,...,x_d,1)
        let mut next_layer_claimed_values = F::ZERO;
        for c in 0..claimed_values.len() / 2 {
            next_layer_claimed_values += random_coeff[c]
                * ((F::ONE - r) * claimed_values[2 * c] + r * claimed_values[2 * c + 1])
        }
        binding_per_layer = next_layer_claimed_values;
    }
    initial_random_point.reverse();
    (
        binding_per_layer,
        random_coeff,
        initial_random_point,
        final_evaluations,
    )
}
