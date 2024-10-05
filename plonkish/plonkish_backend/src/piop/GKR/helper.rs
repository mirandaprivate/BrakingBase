use ff::PrimeField;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use serde::{de::DeserializeOwned, Serialize};

use crate::util::transcript::FieldTranscriptRead;

pub fn evaluate_eq<F: PrimeField>(r_x: &Vec<F>, r_y: &Vec<F>) -> F {
    let mut temp = F::ONE;
    assert_eq!(r_x.len(), r_y.len());
    for k in 0..r_y.len() {
        temp = temp * ((r_x[k] * r_y[k]) + ((F::ONE - r_x[k]) * (F::ONE - r_y[k])));
    }
    temp
}

// CODE  for evaluating polynomial at points
//.............
pub fn eval<F: PrimeField>(p: &[F], x: F) -> F {
    // Horner evaluation
    p.iter()
        .rev()
        .fold(F::ZERO, |acc, &coeff| (acc * x) + coeff)
}

pub fn fold_by_msb<F: PrimeField>(poly: &Vec<F>, point: F) -> Vec<F> {
    let halfsize = poly.len() >> 1;
    let mut res = vec![F::ZERO; halfsize];
    for k in 0..halfsize {
        res[k] = poly[k] + (poly[k + halfsize] - poly[k]) * point;
    }
    res
}

pub fn par_fold_by_msb<F: PrimeField>(poly: &Vec<F>, point: F) -> Vec<F> {
    let halfsize = poly.len() >> 1;
    let mut res = vec![F::ZERO; halfsize];
    res.par_iter_mut().enumerate().for_each(|(j, res_j)| {
        *res_j = poly[j] + (poly[j + halfsize] - poly[j]) * point;
    });
    res
}

pub fn compute_fourier_bases<F: PrimeField>(r: Vec<F>) -> Vec<F> {
    //Initialize fc_eq with (1- r[0]) and r[0]
    let mut fc_eq = [F::ONE - r[r.len() - 1], r[r.len() - 1]].to_vec();
    //Iterate over the length of the r vector
    for k in (0..r.len() - 1).rev() {
        let temp = fc_eq;
        //initialize fc_eq of double size with zero
        fc_eq = vec![F::ZERO; temp.len() * 2];

        if k < 8 {
            for iter in 0..temp.len() {
                fc_eq[2 * iter + 1] = temp[iter] * r[k];
                fc_eq[2 * iter] = temp[iter] - fc_eq[2 * iter + 1];
            }
        } else {
            fc_eq
                .par_chunks_mut(2)
                .zip(temp)
                .for_each(|(fc_eq_pair, temp)| {
                    fc_eq_pair[1] = temp * (r[k as usize]);
                    fc_eq_pair[0] = temp - fc_eq_pair[1];
                })
        }
    }
    fc_eq
}

pub fn len_4_interpolate<F: PrimeField>(evaluations: &mut [F; 4]) {
    let t0 =
        F::from(2).invert().unwrap() * (evaluations[1] + evaluations[2] - evaluations[0].double());
    let t1 = evaluations[1] - evaluations[2] + evaluations[0] + t0.double().double();
    let t2 = F::from(6).invert().unwrap() * (evaluations[3] - t1);
    *evaluations = [
        evaluations[0],
        evaluations[1] - (evaluations[0] + t0 + t2),
        t0,
        t2,
    ]
}
pub fn input_layer_check1<F: PrimeField + Serialize + DeserializeOwned>(
    gamma_tau: &Vec<F>,
    r_x: &Vec<F>,
    r_y: &Vec<F>,
    combiners: &Vec<F>,
    random_points: &Vec<F>,
    expected_eval: F,
    n_circuits: usize,
    transcript: &mut impl FieldTranscriptRead<F>,
) {
    
        let indices_eval = evaluate_indicies::<F>(random_points);
        let mut random_points = random_points.clone();
        random_points.reverse();
        let r_x_eval = evaluate_eq::<F>(r_x, &random_points);
        let r_y_eval = evaluate_eq::<F>(r_y, &random_points);
        let gamma_square = gamma_tau[0].square();
        let final_ts_eval_row = transcript.read_field_element().unwrap();
        let final_ts_eval_col = transcript.read_field_element().unwrap();
        let mut circuit_evals = vec![F::ZERO; n_circuits];
        circuit_evals[0] = indices_eval 
        + gamma_tau[0] * r_x_eval - gamma_tau[1];
        circuit_evals[1] = circuit_evals[0] + gamma_square * final_ts_eval_row;
        circuit_evals[2] = indices_eval + gamma_tau[0] * r_y_eval - gamma_tau[1];
        circuit_evals[3] = circuit_evals[2] + gamma_square * final_ts_eval_col;
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
    transcript: &mut impl FieldTranscriptRead<F>,
) {
    let h_row_eval = transcript.read_field_element().unwrap();
    let h_col_eval = transcript.read_field_element().unwrap();
    let read_ts_row_eval = transcript.read_field_element().unwrap();
    let read_ts_col_eval = transcript.read_field_element().unwrap();
    let h_erow_eval = transcript.read_field_element().unwrap();
    let h_ecol_eval = transcript.read_field_element().unwrap();
    let gamma_square = gamma_tau[0].square();
    let mut circuit_evals = vec![F::ZERO; n_circuits];
    circuit_evals[1] =
        h_row_eval + gamma_tau[0] * h_erow_eval + gamma_square * read_ts_row_eval - gamma_tau[1];
    circuit_evals[0] = circuit_evals[1] + gamma_square;
    circuit_evals[3] =
        h_col_eval + gamma_tau[0] * h_ecol_eval + gamma_square * read_ts_col_eval - gamma_tau[1];
    circuit_evals[2] = circuit_evals[3] + gamma_square;

    let mut final_claimed_values = F::ZERO;
    for c in 0..n_circuits {
        final_claimed_values += combiners[c] * circuit_evals[c]
    }
    assert_eq!(
        expected_eval, final_claimed_values,
        "Input layer check failed of second circuit"
    )
}
pub fn evaluate_indicies<F: PrimeField>(random_values: &Vec<F>) -> F {
    let mut evaluation = F::ZERO;
    for i in 0..random_values.len() {
        evaluation += F::from(1u64 << i) * random_values[i];
    }
    evaluation
}
