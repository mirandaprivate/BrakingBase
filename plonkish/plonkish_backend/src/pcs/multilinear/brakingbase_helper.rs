use crate::util::{
    arithmetic::div_ceil,
    code::ParityCheckMatrix,
    parallel::{num_threads, parallelize_iter},
};
use ff::PrimeField;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

pub fn evaluate_poly<F: PrimeField>(coeffs: &Vec<F>, point: &Vec<F>) -> F {
    let mut eval = F::ZERO;
    let tensor_point = point_to_tensor(1, point).1;
    for i in 0..tensor_point.len() {
        eval += coeffs[i] * tensor_point[i];
    }
    eval
}
pub fn point_to_tensor<F: PrimeField>(num_rows: usize, point: &[F]) -> (Vec<F>, Vec<F>) {
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(point.len() - (num_rows.ilog2() as usize));
    let t_0 = eq_xy(lo); // switch t_0 and t_1
    let t_1 = eq_xy(hi);
    (t_0, t_1)
}

pub fn partial_evaluate_poly<F: PrimeField>(coeffs: &Vec<F>, point: &Vec<F>, skip: usize) -> F {
    let mut eval = F::ZERO;
    let tensor_point = point_to_tensor(1 << (point.len() - skip), point).0;
    for i in 0..tensor_point.len() {
        eval += coeffs[i] * tensor_point[i];
    }
    eval
}
pub fn eq<F: PrimeField>(mut idx: usize, point: &Vec<F>) -> F {
    let mut res = F::ONE;
    for i in 1..=point.len() {
        let bit = idx - ((idx >> 1) << 1);
        let f_bit = F::try_from(bit as u64).unwrap();
        //assert_ne!(point[point.len() - i], F::ZERO);
        res *=
            f_bit * point[point.len() - i] + (F::ONE - f_bit) * (F::ONE - point[point.len() - i]);
        idx = idx >> 1;
    }
    res
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
pub fn eq_xy<F: PrimeField>(y: &[F]) -> Vec<F> {
    if y.is_empty() {
        return vec![F::ZERO; 1];
    }

    let expand_serial = |next_evals: &mut [F], evals: &[F], y_i: &F| {
        for (next_evals, eval) in next_evals.chunks_mut(2).zip(evals.iter()) {
            next_evals[1] = *eval * y_i;
            next_evals[0] = *eval - &next_evals[1];
        }
    };

    let mut evals = vec![F::ONE];
    for y_i in y.iter() {
        let mut next_evals = vec![F::ZERO; 2 * evals.len()];
        if evals.len() < 32 {
            expand_serial(&mut next_evals, &evals, y_i);
        } else {
            let mut chunk_size = div_ceil(evals.len(), num_threads());
            if chunk_size % 2 == 1 {
                chunk_size += 1;
            }
            parallelize_iter(
                next_evals
                    .chunks_mut(chunk_size)
                    .zip(evals.chunks(chunk_size >> 1)),
                |(next_evals, evals)| expand_serial(next_evals, evals, y_i),
            );
        }
        evals = next_evals;
    }

    evals
}
pub fn point_to_tensor_for_commit<F: PrimeField>(num_rows: usize, point: &[F]) -> (Vec<F>, Vec<F>) {
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at((num_rows.ilog2() as usize));
    let x_0 = eq_xy(hi);
    let x_1 = eq_xy(lo);
    (x_0, x_1)
}
pub fn len_3_interpolate<F: PrimeField>(eval: &mut Vec<F>) {
    let t0 = eval[0] - eval[1].double();
    let half = F::from(2 as u64).invert().unwrap();
    eval[1] = (-(eval[0] + eval[2]) - t0.double()) * half;
    eval[2] = (t0 + eval[2]) * half;
}
// CODE  for evaluating polynomial at points
//.............
pub fn eval<F: PrimeField>(p: &[F], x: F) -> F {
    // Horner evaluation
    p.iter()
        .rev()
        .fold(F::ZERO, |acc, &coeff| (acc * x) + coeff)
}

pub fn evaluate_eq<F: PrimeField>(r_x: &Vec<F>, r_y: &Vec<F>) -> F {
    let mut temp = F::ONE;
    assert_eq!(r_x.len(), r_y.len());
    for k in 0..r_y.len() {
        temp = temp * ((r_x[k] * r_y[k]) + ((F::ONE - r_x[k]) * (F::ONE - r_y[k])));
    }
    temp
}
