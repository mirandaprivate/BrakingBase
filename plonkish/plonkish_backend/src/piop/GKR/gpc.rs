use crate::pcs::multilinear::brakingbase_helper::eq;
use ff::PrimeField;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
pub fn grand_product_circuits<F: PrimeField>(
    len1: usize,
    basefold_poly_size: usize,
    h: &Vec<F>,
    h_erow_ecol: &Vec<F>,
    read_ts: &Vec<F>,
    final_ts: &Vec<F>,
    eq_data: &Vec<F>,
    gamma_tau: &Vec<F>,
) -> (Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>) {
    assert!(len1.is_power_of_two(), "len1 must be power of 2");
    assert!(
        basefold_poly_size.is_power_of_two(),
        "basefold_poly_size must be power of 2"
    );

    let depth1 = len1.trailing_zeros() as usize;
    let depth2 = basefold_poly_size.trailing_zeros() as usize;
    let gamma_square = gamma_tau[0].square();
    let ((w_init_circuit_layers, s_circuit_layers), (w_update_circuit_layers, r_circuit_layers)) =
        rayon::join(
            || {
                let mut w_init_circuit_layers = Vec::new();
                let mut s_circuit_layers = Vec::new();
                let (w_init, s): (Vec<F>, Vec<F>) = (0..len1)
                    .into_par_iter()
                    .map(|i| {
                        let term =
                            F::from_u128(i as u128) + gamma_tau[0] * eq(i, &eq_data) - gamma_tau[1];
                        (term, term + gamma_square * final_ts[i])
                    })
                    .collect();
                w_init_circuit_layers.push(w_init);
                s_circuit_layers.push(s);

                (1..depth1 + 1).for_each(|k| {
                    let layer_size = 1 << (depth1 - k);
                    let (temp_w_init, temp_s): (Vec<F>, Vec<F>) = (0..layer_size)
                        .into_par_iter()
                        .map(|i| {
                            (
                                w_init_circuit_layers[k - 1][2 * i]
                                    * w_init_circuit_layers[k - 1][2 * i + 1],
                                s_circuit_layers[k - 1][2 * i] * s_circuit_layers[k - 1][2 * i + 1],
                            )
                        })
                        .collect();
                    w_init_circuit_layers.push(temp_w_init);
                    s_circuit_layers.push(temp_s);
                });
                (w_init_circuit_layers, s_circuit_layers)
            },
            || {
                let (w_update, r): (Vec<F>, Vec<F>) = (0..basefold_poly_size)
                    .into_par_iter()
                    .map(|i| {
                        let term = h[i] + gamma_tau[0] * h_erow_ecol[i] + gamma_square * read_ts[i]
                            - gamma_tau[1];
                        (term + gamma_square, term)
                    })
                    .collect();

                let mut r_circuit_layers = Vec::new();
                let mut w_update_circuit_layers = Vec::new();

                w_update_circuit_layers.push(w_update);
                r_circuit_layers.push(r);

                (1..depth2 + 1).for_each(|k| {
                    let layer_size = 1 << (depth2 - k);
                    let (temp_w_update, temp_r): (Vec<F>, Vec<F>) = (0..layer_size)
                        .into_par_iter()
                        .map(|i| {
                            (
                                w_update_circuit_layers[k - 1][2 * i]
                                    * w_update_circuit_layers[k - 1][2 * i + 1],
                                r_circuit_layers[k - 1][2 * i] * r_circuit_layers[k - 1][2 * i + 1],
                            )
                        })
                        .collect();
                    w_update_circuit_layers.push(temp_w_update);
                    r_circuit_layers.push(temp_r);
                });
                (w_update_circuit_layers, r_circuit_layers)
            },
        );
    assert_eq!(
        w_init_circuit_layers[w_init_circuit_layers.len() - 1][0]
            * w_update_circuit_layers[w_update_circuit_layers.len() - 1][0],
        s_circuit_layers[s_circuit_layers.len() - 1][0]
            * r_circuit_layers[r_circuit_layers.len() - 1][0],
        "Incorrect circuits"
    );
    (
        w_init_circuit_layers,
        w_update_circuit_layers,
        s_circuit_layers,
        r_circuit_layers,
    )
}
