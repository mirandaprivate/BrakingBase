use super::helper::SparseMetaData;
use super::sum_check::matrix_eval_sum_check;
use crate::pcs::multilinear::brakingbase::{batch_sum_check_prover, BrakingbaseSpec};
use crate::pcs::multilinear::brakingbase_helper::{evaluate_poly, point_to_tensor};
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
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{de::DeserializeOwned, Serialize};

pub fn batch_eval_proof<F, H, S>(
    sparse_metadata: Vec<SparseMetaData<F>>,
    mut rx: Vec<F>,
    mut ry: Vec<F>,
    rx_basis_evals: Vec<F>,
    E: &MultilinearPolynomial<F>,
    W: &MultilinearPolynomial<F>,
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
    let ry_basis_evals = point_to_tensor(1, &ry.clone()).1;

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

    let val: Vec<Vec<F>> = sparse_metadata
        .iter()
        .map(|metadata| metadata.val.clone().into_evals())
        .collect();

    let be_sc_rp =
        matrix_eval_sum_check::<F, H, S>(val.clone(), e_rx.clone(), e_ry.clone(), transcript);

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

    let mut random_points1 = gkr_prover::<F, H, S>(
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

    let (
        (rows_evals, cols_evals, read_ts_for_rows_evals, read_ts_for_cols_evals),
        (e_rx_evals, e_ry_evals, final_ts_for_rows_evals, final_ts_for_cols_evals),
    ) = rayon::join(
        || {
            (
                rows.iter()
                    .map(|poly| evaluate_poly(poly, &random_points2))
                    .collect::<Vec<F>>(),
                cols.iter()
                    .map(|poly| evaluate_poly(poly, &random_points2))
                    .collect::<Vec<F>>(),
                read_ts_for_rows
                    .iter()
                    .map(|poly| evaluate_poly(poly, &random_points2))
                    .collect::<Vec<F>>(),
                read_ts_for_cols
                    .iter()
                    .map(|poly| evaluate_poly(poly, &random_points2))
                    .collect::<Vec<F>>(),
            )
        },
        || {
            (
                e_rx.iter()
                    .map(|poly| evaluate_poly(poly, &random_points2))
                    .collect::<Vec<F>>(),
                e_ry.iter()
                    .map(|poly| evaluate_poly(poly, &random_points2))
                    .collect::<Vec<F>>(),
                final_ts_for_rows
                    .iter()
                    .map(|poly| evaluate_poly(poly, &random_points1))
                    .collect::<Vec<F>>(),
                final_ts_for_cols
                    .iter()
                    .map(|poly| evaluate_poly(poly, &random_points1))
                    .collect::<Vec<F>>(),
            )
        },
    );

    transcript.write_field_elements(&rows_evals).unwrap();
    transcript.write_field_elements(&cols_evals).unwrap();
    transcript
        .write_field_elements(&read_ts_for_rows_evals)
        .unwrap();
    transcript
        .write_field_elements(&read_ts_for_cols_evals)
        .unwrap();
    transcript.write_field_elements(&e_rx_evals).unwrap();
    transcript.write_field_elements(&e_ry_evals).unwrap();
    transcript
        .write_field_elements(&final_ts_for_rows_evals)
        .unwrap();
    transcript
        .write_field_elements(&final_ts_for_cols_evals)
        .unwrap();

    let batch_sc_rc = transcript.squeeze_challenges(35);
    let (mut poly1, mut poly2): (Vec<F>, Vec<F>) = (0..E.evals().len())
        .into_par_iter()
        .map(|idx| {
            (
                E.evals()[idx] * batch_sc_rc[0],
                W.evals()[idx] * batch_sc_rc[1],
            )
        })
        .collect();

    let vec2 = batch_sc_rc.iter().skip(2).take(9).collect::<Vec<&F>>();
    let poly3 = (0..e_rx[0].len())
        .into_par_iter()
        .map(|idx| {
            let vec1 = vec![
                e_rx[0][idx],
                e_rx[1][idx],
                e_rx[2][idx],
                e_ry[0][idx],
                e_ry[1][idx],
                e_ry[2][idx],
                val[0][idx],
                val[1][idx],
                val[2][idx],
            ];
            vec1.iter()
                .zip(vec2.iter())
                .fold(F::ZERO, |acc, (value1, value2)| acc + (*value1 * *value2))
        })
        .collect::<Vec<F>>();

    let vec2 = batch_sc_rc.iter().skip(11).take(18).collect::<Vec<&F>>();
    let poly4 = (0..e_rx[0].len())
        .into_par_iter()
        .map(|idx| {
            let vec1 = vec![
                e_rx[0][idx],
                e_rx[1][idx],
                e_rx[2][idx],
                e_ry[0][idx],
                e_ry[1][idx],
                e_ry[2][idx],
                rows[0][idx],
                rows[1][idx],
                rows[2][idx],
                cols[0][idx],
                cols[1][idx],
                cols[2][idx],
                read_ts_for_rows[0][idx],
                read_ts_for_rows[1][idx],
                read_ts_for_rows[2][idx],
                read_ts_for_cols[0][idx],
                read_ts_for_cols[1][idx],
                read_ts_for_cols[2][idx],
            ];
            vec1.iter()
                .zip(vec2.iter())
                .fold(F::ZERO, |acc, (value1, value2)| acc + (*value1 * *value2))
        })
        .collect::<Vec<F>>();

    let vec2 = batch_sc_rc.iter().skip(29).take(6).collect::<Vec<&F>>();
    let mut poly5 = (0..final_ts_for_rows[0].len())
        .into_par_iter()
        .map(|idx| {
            let vec1 = vec![
                final_ts_for_rows[0][idx],
                final_ts_for_rows[1][idx],
                final_ts_for_rows[2][idx],
                final_ts_for_cols[0][idx],
                final_ts_for_cols[1][idx],
                final_ts_for_cols[2][idx],
            ];
            vec1.iter()
                .zip(vec2.iter())
                .fold(F::ZERO, |acc, (value1, value2)| acc + (*value1 * *value2))
        })
        .collect::<Vec<F>>();
    let num_var_witness = poly1.len().trailing_zeros() as usize;
    extend_if_required(
        poly3.len(),
        &mut poly1,
        &mut poly2,
        &mut poly5,
        &mut random_points1,
        &mut rx,
        &mut ry,
    );
    let ((eq_random_points1, eq_random_points2), (eq_be_sc_rp, rx_basis_evals, ry_basis_evals)) =
        rayon::join(
            || {
                (
                    point_to_tensor(1, &random_points1).1,
                    point_to_tensor(1, &random_points2).1,
                )
            },
            || {
                (
                    point_to_tensor(1, &be_sc_rp).1,
                    point_to_tensor(1, &rx).1,
                    point_to_tensor(1, &ry).1,
                )
            },
        );
    let mut polys = Vec::new();
    polys.push(poly1);
    polys.push(poly2);
    polys.push(poly3);
    polys.push(poly4);
    polys.push(poly5);

    let mut eqs = Vec::new();
    eqs.push(rx_basis_evals);
    eqs.push(ry_basis_evals);
    eqs.push(eq_be_sc_rp);
    eqs.push(eq_random_points2);
    eqs.push(eq_random_points1);

    let (_, batch_sum_check_rp) = batch_sum_check_prover::<F, H, S>(&mut polys, eqs, transcript);

    let (
        (
            rows_evals,
            cols_evals,
            read_ts_for_rows_evals,
            read_ts_for_cols_evals,
            e_rx_evals,
            e_ry_evals,
        ),
        (final_ts_for_rows_evals, final_ts_for_cols_evals, val_evals, E_eval, W_eval),
    ) = rayon::join(
        || {
            (
                rows.iter()
                    .map(|poly| evaluate_poly(&poly, &batch_sum_check_rp))
                    .collect::<Vec<F>>(),
                cols.iter()
                    .map(|poly| evaluate_poly(&poly, &batch_sum_check_rp))
                    .collect::<Vec<F>>(),
                read_ts_for_rows
                    .iter()
                    .map(|poly| evaluate_poly(&poly, &batch_sum_check_rp))
                    .collect::<Vec<F>>(),
                read_ts_for_cols
                    .iter()
                    .map(|poly| evaluate_poly(&poly, &batch_sum_check_rp))
                    .collect::<Vec<F>>(),
                e_rx.iter()
                    .map(|poly| evaluate_poly(&poly, &batch_sum_check_rp))
                    .collect::<Vec<F>>(),
                e_ry.iter()
                    .map(|poly| evaluate_poly(&poly, &batch_sum_check_rp))
                    .collect::<Vec<F>>(),
            )
        },
        || {
            (
                final_ts_for_rows
                    .iter()
                    .map(|poly| {
                        evaluate_poly(
                            &poly,
                            &batch_sum_check_rp[batch_sum_check_rp.len() - num_var_witness..]
                                .to_vec(),
                        )
                    })
                    .collect::<Vec<F>>(),
                final_ts_for_cols
                    .iter()
                    .map(|poly| {
                        evaluate_poly(
                            &poly,
                            &batch_sum_check_rp[batch_sum_check_rp.len() - num_var_witness..]
                                .to_vec(),
                        )
                    })
                    .collect::<Vec<F>>(),
                val.iter()
                    .map(|poly| evaluate_poly(&poly, &batch_sum_check_rp))
                    .collect::<Vec<F>>(),
                evaluate_poly(
                    &E.evals().to_vec(),
                    &batch_sum_check_rp[batch_sum_check_rp.len() - num_var_witness..].to_vec(),
                ),
                evaluate_poly(
                    &W.evals().to_vec(),
                    &batch_sum_check_rp[batch_sum_check_rp.len() - num_var_witness..].to_vec(),
                ),
            )
        },
    );
    transcript.write_field_elements(&rows_evals).unwrap();
    transcript.write_field_elements(&cols_evals).unwrap();
    transcript
        .write_field_elements(&read_ts_for_rows_evals)
        .unwrap();
    transcript
        .write_field_elements(&read_ts_for_cols_evals)
        .unwrap();
    transcript.write_field_elements(&e_rx_evals).unwrap();
    transcript.write_field_elements(&e_ry_evals).unwrap();
    transcript
        .write_field_elements(&final_ts_for_rows_evals)
        .unwrap();
    transcript
        .write_field_elements(&final_ts_for_cols_evals)
        .unwrap();
    transcript.write_field_elements(&val_evals).unwrap();
    transcript.write_field_element(&E_eval).unwrap();
    transcript.write_field_element(&W_eval).unwrap();
}

fn extend_if_required<F: PrimeField + Serialize + DeserializeOwned>(
    max_len: usize,
    poly1: &mut Vec<F>,
    poly2: &mut Vec<F>,
    poly5: &mut Vec<F>,
    random_points1: &mut Vec<F>,
    rx: &mut Vec<F>,
    ry: &mut Vec<F>,
) {
    let mut poly_len = poly1.len();
    let count = (max_len.trailing_zeros() - poly_len.trailing_zeros()) as usize;

    if poly_len != max_len {
        let temp1 = poly1.clone();
        let temp2 = poly2.clone();
        let temp3 = poly5.clone();
        let temp4 = random_points1.clone();
        let temp5 = rx.clone();
        let temp6 = ry.clone();

        while poly_len != max_len {
            poly1.extend(temp1.clone());
            poly2.extend(temp2.clone());
            poly5.extend(temp3.clone());
            poly_len = poly1.len();
        }
        random_points1.reverse();
        rx.reverse();
        ry.reverse();

        random_points1.extend(temp4[0..count].to_vec());
        rx.extend(temp5[0..count].to_vec());
        ry.extend(temp6[0..count].to_vec());

        random_points1.reverse();
        rx.reverse();
        ry.reverse();
    }
}
