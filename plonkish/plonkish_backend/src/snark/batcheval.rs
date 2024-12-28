use super::helper::SparseMetaData;
use crate::pcs::multilinear::brakingbase::BrakingbaseSpec;
use crate::pcs::multilinear::brakingbase_helper::{par_fold_by_msb, point_to_tensor};
use crate::piop::GKR::gkr::gkr_prover;
use crate::piop::GKR::gpc::grand_product_circuits;
use crate::poly::multilinear::MultilinearPolynomial;
use crate::poly::Polynomial;
use crate::util::hash::Hash;
use crate::{
    pcs::{
        multilinear::brakingbase::{Brakingbase, BrakingbaseProverParams},
        PolynomialCommitmentScheme,
    },
    util::transcript::TranscriptWrite,
};
use ff::PrimeField;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{de::DeserializeOwned, Serialize};

pub fn batch_eval_proof<F, H, S>(
    sparse_metadata: Vec<SparseMetaData<F>>,
    eval_point: &Vec<F>,
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
    let (rx, ry) = eval_point.split_at(eval_point.len() / 2);
    // let rx_basis_evals = compute_coeff(&rx.to_vec());
    // let ry_basis_evals = compute_coeff(&ry.to_vec());
    let rx_basis_evals = point_to_tensor(1, &rx.to_vec()).1;
    let ry_basis_evals = point_to_tensor(1, &ry.to_vec()).1;

    let e_rx: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| {
            metadata
                .row
                .clone()
                .into_evals()
                .par_iter()
                .map(|row_idx| {
                    let mut bytes = [0; size_of::<u32>()];
                    bytes.copy_from_slice(&row_idx.to_repr().as_ref()[..size_of::<u32>()]);
                    let row_idx = u32::from_le_bytes(bytes) as usize;
                    rx_basis_evals[row_idx]
                })
                .collect()
        })
        .collect();
    let e_ry: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| {
            metadata
                .col
                .clone()
                .into_evals()
                .par_iter()
                .map(|col_idx| {
                    let mut bytes = [0; size_of::<u32>()];
                    bytes.copy_from_slice(&col_idx.to_repr().as_ref()[..size_of::<u32>()]);
                    let col_idx = u32::from_le_bytes(bytes) as usize;
                    ry_basis_evals[col_idx]
                })
                .collect()
        })
        .collect();

    // let e_rx_polys: Vec<Vec<F>> = e_rx.iter().map(|rx| rx.to_vec()).collect();
    // let e_ry_polys: Vec<Vec<F>> = e_ry.iter().map(|ry| ry.to_vec()).collect();

    // let e_rx_polys: Vec<&Vec<F>> = e_rx.iter().collect();
    // let e_ry_polys: Vec<&Vec<F>> = e_ry.iter().collect();

    e_rx.iter().for_each(|rx_poly| {
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp,
            &MultilinearPolynomial::new(rx_poly.to_vec()),
            transcript,
        )
        .unwrap();
    });

    e_ry.iter().for_each(|ry_poly| {
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::commit_and_write(
            pp,
            &MultilinearPolynomial::new(ry_poly.to_vec()),
            transcript,
        )
        .unwrap();
    });

    // let val: Vec<&Vec<F>> = sparse_metadata
    //     .iter()
    //     .map(|metadata| metadata.val.as_coeffs())
    //     .collect();

    let batch_eval_sum_check_rp =
        batch_eval_sum_check::<F, H, S>(&sparse_metadata, e_rx.clone(), e_ry.clone(), transcript);

    // let e_rx_evals_sum_check: Vec<F> = sum_check_transcript
    //     .e_rx
    //     .iter()
    //     .map(|poly| poly.get_coeff(0))
    //     .collect();
    // let e_ry_evals_sum_check: Vec<F> = sum_check_transcript
    //     .e_ry
    //     .iter()
    //     .map(|poly| poly.get_coeff(0))
    //     .collect();
    // let val_evals_sum_check: Vec<F> = sum_check_transcript
    //     .val
    //     .iter()
    //     .map(|poly| poly.get_coeff(0))
    //     .collect();

    // let rx_basis_evals = MultilinearPolynomial<F>::new(rx_basis_evals);
    // let ry_basis_evals = MultPolynomial::new(ry_basis_evals);

    let rows: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| metadata.row.evals().to_vec())
        .collect();
    let cols: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| metadata.col.evals().to_vec())
        .collect();
    let read_ts_for_rows: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| metadata.timestamps.read_ts_row.evals().to_vec())
        .collect();
    let read_ts_for_cols: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| metadata.timestamps.read_ts_col.evals().to_vec())
        .collect();
    let final_ts_for_rows: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| metadata.timestamps.final_ts_row.evals().to_vec())
        .collect();
    let final_ts_for_cols: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| metadata.timestamps.final_ts_col.evals().to_vec())
        .collect();

    let gamma_tau = transcript.squeeze_challenges(2);
    let (
        (
            w_init_circuit_layers_row,
            w_update_circuit_layers_row,
            s_circuit_layers_row,
            r_circuit_layers_row,
        ),
        (
            w_init_circuit_layers_col,
            w_update_circuit_layers_col,
            s_circuit_layers_col,
            r_circuit_layers_col,
        ),
    ) = rayon::join(
        || {
            grand_product_circuits::<F>(
                rx_basis_evals.len(),
                rows[0].len(),
                &rows,
                &e_rx,
                &read_ts_for_rows,
                &final_ts_for_rows,
                &rx_basis_evals,
                &gamma_tau,
            )
        },
        || {
            grand_product_circuits::<F>(
                ry_basis_evals.len(),
                cols[0].len(),
                &cols,
                &e_ry,
                &read_ts_for_cols,
                &final_ts_for_cols,
                &ry_basis_evals,
                &gamma_tau,
            )
        },
    );

    let circuit2_depth = rows[0].len().trailing_zeros() as usize;

    let random_points1 = gkr_prover::<F, H, S>(
        &w_init_circuit_layers_row
            .iter()
            .chain(s_circuit_layers_row.iter())
            .chain(w_init_circuit_layers_col.iter())
            .chain(s_circuit_layers_col.iter())
            .collect(),
        transcript,
    );

    let random_points2 = gkr_prover::<F, H, S>(
        &w_update_circuit_layers_row
            .iter()
            .chain(r_circuit_layers_row.iter())
            .chain(w_update_circuit_layers_col.iter())
            .chain(r_circuit_layers_col.iter())
            .collect(),
        transcript,
    );

    // let final_point_basis_evals = compute_coeff(&gkr_transcript1.final_layer_point);

    // let final_ts_evals_row_mem_check: Vec<F> = final_ts_for_rows
    //     .into_iter()
    //     .map(|final_ts| {
    //         final_ts
    //             .par_iter()
    //             .zip(final_point_basis_evals.par_iter())
    //             .map(|(coeff, basis)| *coeff * *basis)
    //             .reduce(|| F::ZERO, |acc, g| acc + g)
    //     })
    //     .collect();

    // let final_ts_evals_col_mem_check: Vec<F> = final_ts_for_cols
    //     .into_iter()
    //     .map(|final_ts| {
    //         final_ts
    //             .par_iter()
    //             .zip(final_point_basis_evals.par_iter())
    //             .map(|(coeff, basis)| *coeff * *basis)
    //             .reduce(|| F::ZERO, |acc, g| acc + g)
    //     })
    //     .collect();

    // let final_point_basis_evals = compute_coeff(&gkr_transcript2.final_layer_point);

    // let row_evals_mem_check: Vec<_> = rows
    //     .iter()
    //     .map(|row| {
    //         row.par_iter()
    //             .zip(final_point_basis_evals.par_iter())
    //             .map(|(coeff, basis)| *coeff * *basis)
    //             .reduce(|| F::ZERO, |acc, g| acc + g)
    //     })
    //     .collect();
    // let col_evals_mem_check: Vec<_> = cols
    //     .iter()
    //     .map(|col| {
    //         col.par_iter()
    //             .zip(final_point_basis_evals.par_iter())
    //             .map(|(coeff, basis)| *coeff * *basis)
    //             .reduce(|| F::ZERO, |acc, g| acc + g)
    //     })
    //     .collect();
    // let read_ts_evals_row_mem_check: Vec<_> = read_ts_for_rows
    //     .iter()
    //     .map(|read_ts_row| {
    //         read_ts_row
    //             .par_iter()
    //             .zip(final_point_basis_evals.par_iter())
    //             .map(|(coeff, basis)| *coeff * *basis)
    //             .reduce(|| F::ZERO, |acc, g| acc + g)
    //     })
    //     .collect();
    // let read_ts_evals_col_mem_check: Vec<_> = read_ts_for_cols
    //     .iter()
    //     .map(|read_ts_col| {
    //         read_ts_col
    //             .par_iter()
    //             .zip(final_point_basis_evals.par_iter())
    //             .map(|(coeff, basis)| *coeff * *basis)
    //             .reduce(|| F::ZERO, |acc, g| acc + g)
    //     })
    //     .collect();
    // let e_rx_evals_mem_check: Vec<_> = e_rx_polys
    //     .iter()
    //     .map(|e_rx| {
    //         e_rx.par_iter()
    //             .zip(final_point_basis_evals.par_iter())
    //             .map(|(coeff, basis)| *coeff * *basis)
    //             .reduce(|| F::ZERO, |acc, g| acc + g)
    //     })
    //     .collect();
    // let e_ry_evals_mem_check: Vec<_> = e_ry_polys
    //     .iter()
    //     .map(|e_ry| {
    //         e_ry.par_iter()
    //             .zip(final_point_basis_evals.par_iter())
    //             .map(|(coeff, basis)| *coeff * *basis)
    //             .reduce(|| F::ZERO, |acc, g| acc + g)
    //     })
    //     .collect();
    // let batch_eval_coeffs_sum_check = channel.get_random_points(3 * sparse_metadata.len());

    // let sum_check_eval_proof = batch_eval(
    //     &[e_rx_refs, e_ry_refs, val].concat(),
    //     &[
    //         e_rx_evals_sum_check.clone(),
    //         e_ry_evals_sum_check.clone(),
    //         val_evals_sum_check.clone(),
    //     ]
    //     .concat(),
    //     &sum_check_transcript.random_points,
    //     &batch_eval_coeffs_sum_check,
    //     srs,
    // );

    // let batch_eval_coeffs_gkr = channel.get_random_points(8 * sparse_metadata.len());

    // let rows: Vec<&Vec<F>> = sparse_metadata
    //     .iter()
    //     .map(|metadata| metadata.row.as_coeffs())
    //     .collect();
    // let cols: Vec<&Vec<F>> = sparse_metadata
    //     .iter()
    //     .map(|metadata| metadata.col.as_coeffs())
    //     .collect();
    // let read_ts_for_rows: Vec<&Vec<F>> = sparse_metadata
    //     .iter()
    //     .map(|metadata| metadata.timestamps.read_ts_row.as_coeffs())
    //     .collect();
    // let read_ts_for_cols: Vec<&Vec<F>> = sparse_metadata
    //     .iter()
    //     .map(|metadata| metadata.timestamps.read_ts_col.as_coeffs())
    //     .collect();
    // let final_ts_for_rows: Vec<&Vec<F>> = sparse_metadata
    //     .iter()
    //     .map(|metadata| metadata.timestamps.final_ts_row.as_coeffs())
    //     .collect();
    // let final_ts_for_cols: Vec<&Vec<F>> = sparse_metadata
    //     .iter()
    //     .map(|metadata| metadata.timestamps.final_ts_col.as_coeffs())
    //     .collect();

    // let e_rx_refs: Vec<&Vec<F>> = e_rx.iter().map(|rx| rx).collect();
    // let e_ry_refs: Vec<&Vec<F>> = e_ry.iter().map(|ry| ry).collect();

    // let gkr_batch_eval_proof1 = batch_eval(
    //     &[final_ts_for_rows, final_ts_for_cols].concat(),
    //     &[
    //         final_ts_evals_row_mem_check.clone(),
    //         final_ts_evals_col_mem_check.clone(),
    //     ]
    //     .concat(),
    //     &gkr_transcript1.final_layer_point,
    //     &batch_eval_coeffs_gkr,
    //     srs,
    // );
    // let gkr_batch_eval_proof2 = batch_eval(
    //     &[
    //         rows,
    //         cols,
    //         read_ts_for_rows,
    //         read_ts_for_cols,
    //         e_rx_refs,
    //         e_ry_refs,
    //     ]
    //     .concat(),
    //     &[
    //         row_evals_mem_check.clone(),
    //         col_evals_mem_check.clone(),
    //         read_ts_evals_row_mem_check.clone(),
    //         read_ts_evals_col_mem_check.clone(),
    //         e_rx_evals_mem_check.clone(),
    //         e_ry_evals_mem_check.clone(),
    //     ]
    //     .concat(),
    //     &gkr_transcript2.final_layer_point,
    //     &batch_eval_coeffs_gkr,
    //     srs,
    // );

    // BatchSparseEvalProof::new(
    //     sum_check_transcript.clone(),
    //     gkr_transcript1,
    //     gkr_transcript2,
    //     e_rx_evals_sum_check,
    //     e_ry_evals_sum_check,
    //     val_evals_sum_check,
    //     sum_check_eval_proof,
    //     e_rx_commits,
    //     e_ry_commits,
    //     final_ts_evals_row_mem_check,
    //     final_ts_evals_col_mem_check,
    //     row_evals_mem_check,
    //     col_evals_mem_check,
    //     read_ts_evals_row_mem_check,
    //     read_ts_evals_col_mem_check,
    //     e_rx_evals_mem_check,
    //     e_ry_evals_mem_check,
    //     gkr_batch_eval_proof1,
    //     gkr_batch_eval_proof2,
    //     circuit2_depth,
    // )
}

//TODO:- Convert sum check from msb to lsb form
pub fn batch_eval_sum_check<F, H, S>(
    sparse_metadata: &Vec<SparseMetaData<F>>,
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
    let mut val: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| metadata.val.clone().into_evals())
        .collect();

    let sum_check_rounds = val[0].len().trailing_zeros() as usize;

    let random_coeffs = transcript.squeeze_challenges(3);
    // let random_coeffs = channel.get_random_points(3);
    // let mut sum_check_polynomials: Vec<Polynomial> = Vec::new();
    let mut sum_check_random_points: Vec<F> = vec![F::ZERO; sum_check_rounds];

    for i in 0..sum_check_rounds {
        let halfsize = 1usize << (sum_check_rounds - 1usize - i);

        let mut combined_poly = [F::ZERO; 4];
        let mut eval = vec![[F::ZERO; 4]; 3];

        for c in 0..sparse_metadata.len() {
            eval[c] = (0..halfsize).into_iter().fold([F::ZERO; 4], |mut acc, k| {
                acc[0] += val[c][2 * k] * e_rx[c][2 * k] * e_ry[c][2 * k];

                acc[1] += val[c][2 * k + 1] * e_rx[c][2 * k + 1] * e_ry[c][2 * k + 1];

                acc[2] += (val[c][2 * k].double() - val[c][2 * k + 1])
                    * (e_rx[c][2 * k].double() - e_rx[c][2 * k + 1])
                    * (e_ry[c][2 * k].double() - e_ry[c][2 * k + 1]);

                acc[3] += (val[c][2 * k + 1].double() - val[c][2 * k])
                    * (e_rx[c][2 * k + 1].double() - e_rx[c][2 * k])
                    * (e_ry[c][2 * k + 1].double() - e_ry[c][2 * k]);

                acc
            });
            // len_4_interpolate(&mut eval[c])
        }

        for c in 0..sparse_metadata.len() {
            for k in 0..4 {
                combined_poly[k] += random_coeffs[c] * eval[c][k]
            }
        }

        // channel.reseed_with_scalars(&combined_poly);

        // let r_i = channel.get_random_point();
        let r_i = transcript.squeeze_challenge();

        sum_check_random_points[sum_check_rounds - 1 - i] = r_i;

        // sum_check_polynomials.push(Polynomial::new(combined_poly.to_vec()));

        for k in 0..sparse_metadata.len() {
            e_rx[k] = par_fold_by_msb(&e_rx[k], r_i);
            e_ry[k] = par_fold_by_msb(&e_ry[k], r_i);
            val[k] = par_fold_by_msb(&val[k], r_i);
        }
    }
    sum_check_random_points
    // BatchSpartanSumCheckTranscript::new(
    //     sum_check_polynomials,
    //     sum_check_random_points,
    //     e_rx,
    //     e_ry,
    //     val,
    // )
}
