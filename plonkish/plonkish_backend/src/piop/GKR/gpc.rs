use ff::PrimeField;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
pub fn grand_product_circuits<F: PrimeField>(
    len1: usize,
    len2: usize,
    memory_index: &Vec<Vec<F>>, //rows
    h_erow_ecol: &Vec<Vec<F>>,  //e_rx
    read_ts: &Vec<Vec<F>>,
    final_ts: &Vec<Vec<F>>,
    eq_data: &Vec<F>, //rx_basis
    gamma_tau: &Vec<F>,
) -> (
    Vec<Vec<F>>,
    Vec<Vec<Vec<F>>>,
    Vec<Vec<Vec<F>>>,
    Vec<Vec<Vec<F>>>,
) {
    let n_circuits = memory_index.len();
    assert!(len1.is_power_of_two(), "len1 must be power of 2");
    assert!(len2.is_power_of_two(), "len2 must be power of 2");

    let depth1 = len1.trailing_zeros() as usize;
    let depth2 = len2.trailing_zeros() as usize;
    let gamma_square = gamma_tau[0].square();

    let mut w_init_circuit_layers = Vec::new();
    let mut s_circuit_layers = Vec::new();
    let mut w_update_circuit_layers = Vec::new();
    let mut r_circuit_layers = Vec::new();

    //TODO:- Multithread over n_circuits
    let w_init: Vec<F> = (0..len1)
        .into_par_iter()
        .map(|i| F::from_u128(i as u128) + gamma_tau[0] * eq_data[i] - gamma_tau[1])
        .collect();
    w_init_circuit_layers.push(w_init.clone());
    (1..depth1 + 1).for_each(|k| {
        let layer_size = 1 << (depth1 - k);
        let temp_w_init: Vec<F> = (0..layer_size)
            .into_par_iter()
            .map(|i| w_init_circuit_layers[k - 1][2 * i] * w_init_circuit_layers[k - 1][2 * i + 1])
            .collect();
        w_init_circuit_layers.push(temp_w_init);
    });

    for c in 0..n_circuits {
        let ((s_circuit_layer), (w_update_circuit_layer, r_circuit_layer)) = rayon::join(
            || {
                let mut s_circuit_layers = Vec::new();
                let s: Vec<F> = (0..len1)
                    .into_par_iter()
                    .map(|i| w_init[i] + gamma_square * final_ts[c][i])
                    .collect();
                s_circuit_layers.push(s);

                (1..depth1 + 1).for_each(|k| {
                    let layer_size = 1 << (depth1 - k);
                    let temp_s: Vec<F> = (0..layer_size)
                        .into_par_iter()
                        .map(|i| {
                            s_circuit_layers[k - 1][2 * i] * s_circuit_layers[k - 1][2 * i + 1]
                        })
                        .collect();
                    s_circuit_layers.push(temp_s);
                });
                s_circuit_layers
            },
            || {
                let (w_update, r): (Vec<F>, Vec<F>) = (0..len2)
                    .into_par_iter()
                    .map(|i| {
                        let term = memory_index[c][i]
                            + gamma_tau[0] * h_erow_ecol[c][i]
                            + gamma_square * read_ts[c][i]
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

        s_circuit_layers.push(s_circuit_layer);
        w_update_circuit_layers.push(w_update_circuit_layer);
        r_circuit_layers.push(r_circuit_layer);
    }
    (0..n_circuits).for_each(|c| {
        assert_eq!(
            w_init_circuit_layers[w_init_circuit_layers.len() - 1][0]
                * w_update_circuit_layers[c][w_update_circuit_layers[c].len() - 1][0],
            s_circuit_layers[c][s_circuit_layers[c].len() - 1][0]
                * r_circuit_layers[c][r_circuit_layers[c].len() - 1][0],
            "Incorrect circuits"
        );
    });
    (
        w_init_circuit_layers,
        w_update_circuit_layers,
        s_circuit_layers,
        r_circuit_layers,
    )
}
