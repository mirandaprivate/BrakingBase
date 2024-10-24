use super::basefold::{
    Basefold, BasefoldCommitment, BasefoldExtParams, BasefoldParams, BasefoldProverParams,
    BasefoldVerifierParams, Type1Polynomial, Type2Polynomial,
};
use super::brakedown::MultilinearBrakedownCommitment;
use super::brakingbase_helper::{len_3_interpolate, point_to_tensor};
use crate::pcs::multilinear::brakingbase_helper::{
    eq, eq_xy, eval, evaluate_eq, evaluate_poly, par_fold_by_msb, partial_evaluate_poly,
    point_to_tensor_for_commit,
};
use crate::pcs::multilinear::{basefold, brakedown};
use crate::pcs::Commitment;
use crate::piop::sum_check::{self, evaluate};
use crate::piop::sum_check::{
    classic::{ClassicSumCheck, CoefficientsProver},
    eq_xy_eval, SumCheck as _, VirtualPolynomial,
};
use crate::piop::GKR::gkr::{gkr_prover, gkr_verifier};
use crate::piop::GKR::gpc::grand_product_circuits;
use crate::piop::GKR::helper::{input_layer_check1, input_layer_check2};
use crate::util::code::{self, ParityCheckMatrix};
use crate::util::transcript::FieldTranscriptRead;
use crate::{
    pcs::{
        multilinear::{additive, validate_input},
        AdditiveCommitment, Evaluation, Point, PolynomialCommitmentScheme,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{div_ceil, horner, inner_product, steps, BatchInvert, Field, PrimeField},
        code::{Brakedown, BrakedownSpec, LinearCodes},
        expression::{Expression, Query, Rotation},
        hash::{Hash, Output},
        new_fields::{Mersenne127, Mersenne61},
        parallel::{num_threads, parallelize, parallelize_iter},
        transcript::{FieldTranscript, TranscriptRead, TranscriptWrite},
        BigUint, Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use aes::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use bitvec::vec;
use core::fmt::Debug;
use core::ptr::addr_of;
use core::{hash, num, panic};
use ctr;
use ff::{derive, BatchInverter};
use generic_array::GenericArray;
use halo2_curves::bn256::{Bn256, Fr};
use halo2_proofs::circuit::Table;
use halo2_proofs::poly::commitment;
use plonky2_util::{
    ceil_div_usize, log2_strict, reverse_bits, reverse_index_bits, reverse_index_bits_in_place,
};
use rand::random;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha12Rng, ChaCha8Rng,
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    ParallelSlice, ParallelSliceMut,
};
use std::cmp::max;
use std::mem::swap;
use std::{borrow::Cow, marker::PhantomData, mem::size_of, slice};
use std::{collections::HashMap, iter, ops::Deref, time::Instant};

const BLOW_UP_FACTOR: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingbaseParams<F: PrimeField, H: Hash> {
    num_vars: usize,
    brakedown: Brakedown<F>,
    brakedown_num_rows: usize,
    num_brakedown_queries: usize,
    brakedown_row_len: usize,
    brakedown_codeword_len: usize,
    partity_check_matrix: ParityCheckMatrix<F>,
    basefold_poly_size: usize,
    basefold: BasefoldParams<F>,
    basefold_prover_params: BasefoldProverParams<F>,
    basefold_verifier_params: BasefoldVerifierParams<F>,
    trusted_commit: BasefoldBatchCommitment<F, H>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingbaseProverParams<F: PrimeField, H: Hash> {
    num_vars: usize,
    brakedown: Brakedown<F>, // parity check matrix implicitly provided here
    brakedown_num_rows: usize,
    num_brakedown_queries: usize,
    parity_check_matrix: ParityCheckMatrix<F>,
    basefold_poly_size: usize,
    basefold: BasefoldProverParams<F>,
    trusted_commit: BasefoldBatchCommitment<F, H>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingbaseVerifierParams<F: PrimeField, H: Hash> {
    num_vars: usize,
    brakedown_num_rows: usize,
    num_brakedown_queries: usize,
    brakedown_row_len: usize,
    brakedown_codeword_len: usize,
    basefold_poly_size: usize,
    basefold: BasefoldVerifierParams<F>,
    trusted_commit: Output<H>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingbaseCommitment<F: PrimeField, H: Hash> {
    rows: Vec<F>,
    intermediate_hashes: Vec<Output<H>>,
    root: Output<H>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BasefoldBatchCommitment<F: PrimeField, H: Hash> {
    pub codewords: Vec<Type1Polynomial<F>>,
    pub codeword_tree: Vec<Vec<Output<H>>>,
    pub bh_evals: Vec<Type1Polynomial<F>>,
}

impl<F: PrimeField, H: Hash> BrakingbaseProverParams<F, H> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: PrimeField, H: Hash> BrakingbaseCommitment<F, H> {
    fn from_root(root: Output<H>) -> Self {
        Self {
            rows: Vec::new(),
            intermediate_hashes: vec![],
            root,
        }
    }
}

impl<F: PrimeField, H: Hash> AsRef<[Output<H>]> for BrakingbaseCommitment<F, H> {
    fn as_ref(&self) -> &[Output<H>] {
        let root = &self.root;
        slice::from_ref(root)
    }
}

impl<F: PrimeField, H: Hash> AsRef<Output<H>> for BrakingbaseCommitment<F, H> {
    fn as_ref(&self) -> &Output<H> {
        let root = &self.root;
        &root
    }
}

impl<F: PrimeField, H: Hash> Default for BrakingbaseCommitment<F, H> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            intermediate_hashes: vec![Output::<H>::default()],
            root: Output::<H>::default(),
        }
    }
}

pub trait BrakingbaseSpec: BrakedownSpec + BasefoldExtParams + Debug {}

#[derive(Debug)]
pub struct Brakingbase<F: PrimeField, H: Hash, S: BrakingbaseSpec>(PhantomData<(F, H, S)>);

impl<F: PrimeField, H: Hash, S: BrakingbaseSpec> Clone for Brakingbase<F, H, S> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<F, H, S> PolynomialCommitmentScheme<F> for Brakingbase<F, H, S>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    type Param = BrakingbaseParams<F, H>;
    type ProverParam = BrakingbaseProverParams<F, H>;
    type VerifierParam = BrakingbaseVerifierParams<F, H>;
    type Polynomial = MultilinearPolynomial<F>;
    type Commitment = BrakingbaseCommitment<F, H>;
    type CommitmentChunk = Output<H>;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;

        // Generate the Brakedown code.
        let brakedown_num_rows = 2 * num_vars.next_power_of_two();
        let brakedown = Brakedown::new::<S>(
            num_vars,
            (20).min((1 << num_vars) - 1),
            brakedown_num_rows,
            rng,
        );
        let brakedown_row_len = brakedown.row_len();
        let brakedown_codeword_len = brakedown.codeword_len();
        let num_brakedown_queries = brakedown.num_column_opening();
        let parity_check_matrix = brakedown.parity_check_matrix();

        // Generate BaseFold parameters by running BaseFold's setup algo.
        let len = parity_check_matrix.val.len();
        let basefold_poly_size = len.next_power_of_two();
        let mut rng2 = ChaCha8Rng::from_entropy();
        let basefold = Basefold::<F, H, S>::setup(basefold_poly_size, batch_size, rng2).unwrap();

        // Compute the trusted commits
        let (basefold_prover_params, basefold_verifier_params) =
            Basefold::<F, H, S>::trim(&basefold, poly_size, batch_size).unwrap();

        let mut val = parity_check_matrix.clone().val;
        val.resize(basefold_poly_size, F::ZERO);

        let mut row: Vec<F> = parity_check_matrix
            .row
            .par_iter()
            .map(|&elem| F::try_from(elem as u64).unwrap())
            .collect();
        row.resize(basefold_poly_size, row[0]);

        let mut col: Vec<F> = parity_check_matrix
            .col
            .par_iter()
            .map(|&elem| F::try_from(elem as u64).unwrap())
            .collect();
        col.resize(basefold_poly_size, col[0]);

        let (mut read_ts_row, mut final_ts_row, mut read_ts_col, mut final_ts_col) =
            get_timestamps(&row, &col, 2 * brakedown_row_len, len);

        read_ts_row.resize(basefold_poly_size, F::ZERO);
        read_ts_col.resize(basefold_poly_size, F::ZERO);

        let mut final_ts_row_col = Vec::<F>::new();

        assert_eq!(final_ts_row.len(), final_ts_col.len());
        for i in 0..basefold_poly_size / (2 * final_ts_row.len()) {
            final_ts_row_col.extend(final_ts_row.clone());
            final_ts_row_col.extend(final_ts_col.clone());
        }
        let mut polys = Vec::<Vec<F>>::with_capacity(6);
        polys.push(val);
        polys.push(row);
        polys.push(col);
        polys.push(read_ts_row);
        polys.push(read_ts_col);
        polys.push(final_ts_row_col);

        let trusted_commit = basefold_batch_commit::<F, H, S>(&basefold_prover_params, &mut polys);

        Ok(BrakingbaseParams {
            num_vars,
            brakedown,
            brakedown_num_rows,
            num_brakedown_queries,
            brakedown_row_len,
            brakedown_codeword_len,
            partity_check_matrix: parity_check_matrix,
            basefold_poly_size,
            basefold,
            basefold_prover_params,
            basefold_verifier_params,
            trusted_commit,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        Ok((
            BrakingbaseProverParams {
                num_vars: param.num_vars,
                brakedown: param.brakedown.clone(),
                brakedown_num_rows: param.brakedown_num_rows,
                num_brakedown_queries: param.num_brakedown_queries,
                parity_check_matrix: param.partity_check_matrix.clone(),
                basefold_poly_size: param.basefold_poly_size,
                basefold: param.basefold_prover_params.clone(),
                trusted_commit: param.trusted_commit.clone(),
            },
            BrakingbaseVerifierParams {
                num_vars: param.num_vars,
                brakedown_num_rows: param.brakedown_num_rows,
                num_brakedown_queries: param.num_brakedown_queries,
                brakedown_row_len: param.brakedown_row_len,
                brakedown_codeword_len: param.brakedown_codeword_len,
                basefold_poly_size: param.basefold_poly_size,
                basefold: param.basefold_verifier_params.clone(),
                trusted_commit: param
                    .trusted_commit
                    .codeword_tree
                    .last()
                    .unwrap()
                    .last()
                    .unwrap()
                    .clone(),
            },
        ))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        let row_len = pp.brakedown.row_len();

        let codeword_len = pp.brakedown.codeword_len();
        let mut rows = vec![F::ZERO; pp.brakedown_num_rows * codeword_len];

        // Encode rows.
        let encoding_time = Instant::now();
        let chunk_size = div_ceil(pp.brakedown_num_rows, num_threads());
        parallelize_iter(
            rows.chunks_mut(chunk_size * codeword_len)
                .zip(poly.evals().chunks(chunk_size * row_len)), // All elements of row handlled together
            |(rows, evals)| {
                for (row, evals) in rows.chunks_mut(codeword_len).zip(evals.chunks(row_len)) {
                    row[..evals.len()].copy_from_slice(evals);
                    pp.brakedown.encode(row);
                }
            },
        );

        let now = Instant::now();

        // Hash columns
        let depth = codeword_len.next_power_of_two().ilog2() as usize;

        let new_n = Instant::now();
        let mut hashes = vec![Output::<H>::default(); (2 << depth) - 1];

        parallelize(&mut hashes[..codeword_len], |(hashes, start)| {
            let mut hasher = H::new();
            for (hash, column) in hashes.iter_mut().zip(start..) {
                rows.iter()
                    .skip(column)
                    .step_by(codeword_len)
                    .for_each(|item| hasher.update_field_element(item));
                hasher.finalize_into_reset(hash);
            }
        });

        // Merklize column hashes
        let mut offset = 0;
        for width in (1..=depth).rev().map(|depth| 1 << depth) {
            let (input, output) = hashes[offset..].split_at_mut(width);
            //let num_threads = env::var("RAYON_NUM_THREADS").unwrap();
            let chunk_size = div_ceil(output.len(), num_threads());
            parallelize_iter(
                input
                    .chunks(2 * chunk_size)
                    .zip(output.chunks_mut(chunk_size)),
                |(input, output)| {
                    let mut hasher = H::new();

                    for (input, output) in input.chunks_exact(2).zip(output.iter_mut()) {
                        hasher.update(&input[0]);
                        hasher.update(&input[1]);
                        hasher.finalize_into_reset(output);
                    }
                },
            );
            offset += width;
        }

        let (intermediate_hashes, root) = {
            let mut intermediate_hashes = hashes;
            let root = intermediate_hashes.pop().unwrap();
            (intermediate_hashes, root)
        };

        Ok(BrakingbaseCommitment {
            rows,
            intermediate_hashes,
            root,
        })
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error>
    where
        Self::Polynomial: 'a,
    {
        let polys_vec: Vec<&Self::Polynomial> = polys.into_iter().map(|poly| poly).collect();
        polys_vec
            .par_iter()
            .map(|poly| Self::commit(pp, poly))
            .collect()
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        let num_rows = pp.brakedown_num_rows;
        let codeword_len = pp.brakedown.codeword_len();
        let row_len = pp.brakedown.row_len();
        let basefold_poly_size = pp.basefold_poly_size;
        let (mut x_0, mut x_1) = point_to_tensor_for_commit(num_rows, point);
        let mut combined_codeword = vec![F::ZERO; codeword_len];

        // Taking a linear combination of the rows of the commitment matrix
        combined_codeword
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, codeword)| {
                for i in 0..num_rows {
                    *codeword += x_0[i] * comm.rows[codeword_len * i + j];
                }
            });

        // Commiting to the message and (codeword - message) parts of combined_codeword
        let mut p_p_prime: Vec<F> = Vec::new();

        // The number of coefficients in H is pp.blow_up_factor * row_len.
        for i in 0..pp.basefold_poly_size / (2 * row_len) {
            p_p_prime.extend(&combined_codeword);
            p_p_prime.extend(&vec![F::ZERO; 2 * row_len - codeword_len]);
        }

        let p_p_prime_commit = Basefold::<F, H, S>::commit(
            &pp.basefold,
            &MultilinearPolynomial::new(reverse_index_bits(&p_p_prime)),
        )
        .unwrap();
        transcript.write_commitment(p_p_prime_commit.codeword_tree_root());

        // Proximity test for the commitment matrix
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        let mut col_idx = vec![0 as usize; pp.num_brakedown_queries];
        for i in 0..pp.num_brakedown_queries {
            col_idx[i] = squeeze_challenge_idx(transcript, codeword_len);
            transcript
                .write_field_elements(comm.rows.iter().skip(col_idx[i]).step_by(codeword_len))?;

            let mut offset = 0;
            for (idx, width) in (1..=depth).rev().map(|depth| 1 << depth).enumerate() {
                let neighbor_idx = (col_idx[i] >> idx) ^ 1;
                transcript.write_commitment(&comm.intermediate_hashes[offset + neighbor_idx])?;
                offset += width;
            }
        }

        let mut u = transcript.squeeze_challenges(row_len.ilog2().try_into().unwrap());

        //TODO 2: Realise H(X,u) vector, that is, MLE of the matrix H with Y coordinates replaced by u. This is now a polynomial in X variables.
        let mut h = evaluate_H(&pp.parity_check_matrix, &u, pp.brakedown.codeword_len());

        h.resize(2 * row_len, F::ZERO);

        let small_p_p_prime = p_p_prime[0..2 * row_len].to_vec();

        //TODO:- Check if evaluate poly is optimized
        let p_prime_eval_u = &evaluate_poly(&small_p_p_prime[row_len..].to_vec(), &u);
        transcript.write_field_element(&p_prime_eval_u);

        let mut mask = vec![F::ZERO; 2 * row_len];
        let challenges = transcript.squeeze_challenges(pp.num_brakedown_queries);

        for i in 0..pp.num_brakedown_queries {
            mask[col_idx[i]] += challenges[i];
        }

        let random_combiners = transcript.squeeze_challenges(2);

        let sum_check_rounds = (2 * row_len).next_power_of_two().ilog2() as usize;

        //rx
        let mut first_sum_check_random_points = vec![F::ZERO; sum_check_rounds];

        first_sum_check_prover::<F, H, S>(
            sum_check_rounds,
            small_p_p_prime,
            mask,
            h.clone(),
            random_combiners,
            &mut first_sum_check_random_points,
            transcript,
        );

        //TODO 3: evaluate h, p, p_prime at first_sum_check_random_points. Shouldn't folding in the sum-check give this?
        // let h_eval = evaluate_poly(&h, &first_sum_check_random_points);
        let p_eval = partial_evaluate_poly(
            &p_p_prime[0..row_len].to_vec(),
            &first_sum_check_random_points,
            1,
        ); // Suboptimal as to_vec() copies

        let p_prime_eval = partial_evaluate_poly(
            &p_p_prime[row_len..2 * row_len].to_vec(),
            &first_sum_check_random_points,
            1,
        ); // Suboptimal as to_vec() copies
        transcript.write_field_elements([p_eval, p_prime_eval].iter());

        //TODO 6.2(Bhargav): Compute H_val -- Check sum_check_rounds

        let mut h_val = pp.parity_check_matrix.val.clone();
        h_val.resize(basefold_poly_size, F::ZERO);

        // Computing and commiting h_erow and h_ecol
        let mut h_erow =
            compute_oracle_poly(&pp.parity_check_matrix.row, &first_sum_check_random_points);
        h_erow.resize(basefold_poly_size, h_erow[0]);

        //ry
        let mut padded_u = [F::ZERO].to_vec();
        padded_u.extend(&u);

        let mut h_ecol = compute_oracle_poly(&pp.parity_check_matrix.col, &padded_u);
        h_ecol.resize(basefold_poly_size, h_ecol[0]);

        let mut polys: Vec<Vec<F>> = [h_erow.clone(), h_ecol.clone()].to_vec();

        let h_erow_ecol_commit = basefold_batch_commit::<F, H, S>(&pp.basefold, &mut polys);

        let temp = h_erow_ecol_commit.codeword_tree.len();
        transcript.write_commitment(&h_erow_ecol_commit.codeword_tree[temp - 1][0]);

        assert!(h_val.len().is_power_of_two());

        let sum_check_rounds = pp.basefold_poly_size.ilog2() as usize; // Changed by Bhargav
        let mut second_sum_check_random_points = vec![F::ZERO; sum_check_rounds];

        second_sum_check_prover::<F, H, S>(
            sum_check_rounds,
            h_erow.clone(),
            h_ecol.clone(),
            h_val.clone(),
            &mut second_sum_check_random_points,
            transcript,
        );

        //Sample two random points gamma, tau.
        let gamma_tau = transcript.squeeze_challenges(2);

        //let mut h_col = vec![F::ZERO; pp.parity_check_matrix.col.len()];

        let mut h_row: Vec<F> = pp
            .parity_check_matrix
            .row
            .par_iter()
            .map(|&elem| F::from(elem as u64))
            .collect();
        h_row.resize(basefold_poly_size, h_row[0]);

        let mut h_col: Vec<F> = pp
            .parity_check_matrix
            .col
            .par_iter()
            .map(|&elem| F::from(elem as u64))
            .collect();
        h_col.resize(basefold_poly_size, h_col[0]);

        let mut read_ts_row: Vec<F> = pp.trusted_commit.bh_evals[3].poly.to_vec();
        let mut final_ts_row: Vec<F> = pp.trusted_commit.bh_evals[5].poly[0..2 * row_len].to_vec();
        let mut read_ts_col: Vec<F> = pp.trusted_commit.bh_evals[4].poly.to_vec();
        let mut final_ts_col: Vec<F> =
            pp.trusted_commit.bh_evals[5].poly[2 * row_len..4 * row_len].to_vec();
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
                grand_product_circuits(
                    2 * row_len,
                    basefold_poly_size,
                    &h_row,
                    &h_erow,
                    &read_ts_row,
                    &final_ts_row,
                    &first_sum_check_random_points,
                    &gamma_tau,
                )
            },
            || {
                grand_product_circuits::<F>(
                    2 * row_len,
                    basefold_poly_size,
                    &h_col,
                    &h_ecol,
                    &read_ts_col,
                    &final_ts_col,
                    &padded_u,
                    &gamma_tau,
                )
            },
        );

        let random_points1 = gkr_prover::<F, H, S>(
            &[
                &w_init_circuit_layers_row,
                &s_circuit_layers_row,
                &w_init_circuit_layers_col,
                &s_circuit_layers_col,
            ]
            .to_vec(),
            transcript,
        );
        let final_ts_row_eval = evaluate_poly(&final_ts_row, &random_points1);
        let final_ts_col_eval = evaluate_poly(&final_ts_col, &random_points1);
        transcript.write_field_elements(&[final_ts_row_eval, final_ts_col_eval]);

        let random_points2 = gkr_prover::<F, H, S>(
            &[
                &w_update_circuit_layers_row,
                &r_circuit_layers_row,
                &w_update_circuit_layers_col,
                &r_circuit_layers_col,
            ]
            .to_vec(),
            transcript,
        );
        let (
            (h_row_eval, h_col_eval, read_ts_row_eval),
            (read_ts_col_eval, h_erow_eval, h_ecol_eval),
        ) = rayon::join(
            || {
                (
                    evaluate_poly(&h_row, &random_points2),
                    evaluate_poly(&h_col, &random_points2),
                    evaluate_poly(&read_ts_row, &random_points2),
                )
            },
            || {
                (
                    evaluate_poly(&read_ts_col, &random_points2),
                    evaluate_poly(&h_erow, &random_points2),
                    evaluate_poly(&h_ecol, &random_points2),
                )
            },
        );

        transcript.write_field_elements(&[
            h_row_eval,
            h_col_eval,
            read_ts_row_eval,
            read_ts_col_eval,
            h_erow_eval,
            h_ecol_eval,
        ]);

        //Extended random points for p+p' corresponding to first_sum_check_random_points;
        let mut p_p_prime_fsrp = vec![
            F::ZERO;
            second_sum_check_random_points.len()
                - first_sum_check_random_points.len()
        ];
        //Evaluation of p+p' at extended random points corresponding to first_sum_check_random_points;
        p_p_prime_fsrp.append(&mut first_sum_check_random_points);

        //Extended random points for p+p' corresponding to u;
        let mut p_p_prime_rp_u = vec![F::ZERO; second_sum_check_random_points.len() - 1 - u.len()];
        p_p_prime_rp_u.push(F::ONE);
        p_p_prime_rp_u.append(&mut u);
        let p_p_prime_rp_u_eval = evaluate_poly(&p_p_prime, &p_p_prime_rp_u);

        //Extended random points for p+p' corresponding to x0;
        let mut p_p_prime_rp_x0 =
            vec![F::ZERO; second_sum_check_random_points.len() - 1 - x_1.len().ilog2() as usize];
        p_p_prime_rp_x0.push(F::ZERO);

        let mut point_clone = point.to_vec();
        p_p_prime_rp_x0.append(&mut point_clone[(x_0.len().ilog2() as usize)..].to_vec());
        let p_p_prime_rp_x0_eval = evaluate_poly(&p_p_prime, &p_p_prime_rp_x0);

        transcript.write_field_elements(&[p_p_prime_rp_u_eval, p_p_prime_rp_x0_eval]);
        //Append final_ts_row and fianl_ts_row
        let final_ts_row_len = final_ts_row.len(); // can be removed
        let mut final_ts_row_col = [final_ts_row, final_ts_col].concat();

        // Need to sample an extra random point to combine values here
        let r = transcript.squeeze_challenge();

        let mut final_ts_row_col_rp =
            vec![F::ZERO; second_sum_check_random_points.len() - 1 - random_points1.len()];
        final_ts_row_col_rp.push(r);
        final_ts_row_col_rp.extend(random_points1);

        let mut extended_final_ts_row_col = final_ts_row_col.clone();

        //TODO:- Run batch sum check without extending
        for _ in 1..basefold_poly_size / (2 * final_ts_row_len) {
            // denominator can be replaced by 4 * row_len
            extended_final_ts_row_col.extend(final_ts_row_col.clone());
        }

        // evaluations to be batched in total
        let batch_sum_check_random_combiner = transcript.squeeze_challenges(13);

        //Build eq_vector corresponding to each point (6 in total)
        let (
            (eq_p_prime_fsrp, eq_p_prime_rp_u, eq_p_prime_rp_x0),
            (eq_schrp, eq_rp2, eq_final_ts_rc_rp),
        ) = rayon::join(
            || {
                (
                    point_to_tensor(1, &p_p_prime_fsrp).1,
                    point_to_tensor(1, &p_p_prime_rp_u).1,
                    point_to_tensor(1, &p_p_prime_rp_x0).1,
                )
            },
            || {
                (
                    point_to_tensor(1, &second_sum_check_random_points).1,
                    point_to_tensor(1, &random_points2).1,
                    point_to_tensor(1, &final_ts_row_col_rp).1,
                )
            },
        );

        let vec2 = [
            batch_sum_check_random_combiner[0],
            batch_sum_check_random_combiner[1],
            batch_sum_check_random_combiner[2],
        ];
        let poly1 = (0..eq_p_prime_fsrp.len())
            .into_par_iter()
            .map(|idx| {
                let vec1 = vec![
                    eq_p_prime_fsrp[idx],
                    eq_p_prime_rp_u[idx],
                    eq_p_prime_rp_x0[idx],
                ];
                vec1.into_par_iter()
                    .zip(vec2.into_par_iter())
                    .fold_with(F::ZERO, |acc, (value1, value2)| acc + (value1 * value2))
                    .reduce_with(|acc, val| acc + val)
                    .unwrap()
            })
            .collect::<Vec<F>>();

        let vec2 = [
            batch_sum_check_random_combiner[3],
            batch_sum_check_random_combiner[4],
            batch_sum_check_random_combiner[5],
        ];
        let poly2 = (0..h_val.len())
            .into_par_iter()
            .map(|idx| {
                let vec1 = vec![h_val[idx], h_erow[idx], h_ecol[idx]];
                vec1.iter()
                    .zip(vec2.iter())
                    .fold(F::ZERO, |acc, (value1, value2)| acc + (*value1 * *value2))
            })
            .collect::<Vec<F>>();

        let vec2 = [
            batch_sum_check_random_combiner[6],
            batch_sum_check_random_combiner[7],
            batch_sum_check_random_combiner[8],
            batch_sum_check_random_combiner[9],
            batch_sum_check_random_combiner[10],
            batch_sum_check_random_combiner[11],
        ];
        let poly3 = (0..h_erow.len())
            .into_par_iter()
            .map(|idx| {
                let vec1 = vec![
                    h_erow[idx],
                    h_ecol[idx],
                    h_row[idx],
                    h_col[idx],
                    read_ts_row[idx],
                    read_ts_col[idx],
                ];
                vec1.iter()
                    .zip(vec2.iter())
                    .fold(F::ZERO, |acc, (value1, value2)| acc + (*value1 * *value2))
            })
            .collect::<Vec<F>>();

        let mut polys = Vec::new();
        polys.push(poly1);
        polys.push(poly2);
        polys.push(poly3);
        //Multiply with combiner
        let extended_final_ts_row_col1: Vec<F> = extended_final_ts_row_col
            .par_iter()
            .map(|final_ts| *final_ts * batch_sum_check_random_combiner[12])
            .collect();
        polys.push(extended_final_ts_row_col1);

        let mut eqs = Vec::new();
        eqs.push(p_p_prime.clone());
        eqs.push(eq_schrp);
        eqs.push(eq_rp2);
        eqs.push(eq_final_ts_rc_rp);

        let start_time = Instant::now();
        let (p_p_prime_eval, mut batch_sum_check_rp) =
            batch_sum_check_prover::<F, H, S>(&mut polys, eqs, transcript);

        let (
            (h_val_eval, h_erow_eval, h_ecol_eval, h_row_eval),
            (h_col_eval, read_ts_row_eval, read_ts_col_eval, extended_final_ts_row_col_eval),
        ) = rayon::join(
            || {
                (
                    evaluate_poly(&h_val, &batch_sum_check_rp),
                    evaluate_poly(&h_erow, &batch_sum_check_rp),
                    evaluate_poly(&h_ecol, &batch_sum_check_rp),
                    evaluate_poly(&h_row, &batch_sum_check_rp),
                )
            },
            || {
                (
                    evaluate_poly(&h_col, &batch_sum_check_rp),
                    evaluate_poly(&read_ts_row, &batch_sum_check_rp),
                    evaluate_poly(&read_ts_col, &batch_sum_check_rp),
                    evaluate_poly(&extended_final_ts_row_col, &batch_sum_check_rp),
                )
            },
        );

        transcript
            .write_field_elements(&[
                h_val_eval,
                h_erow_eval,
                h_ecol_eval,
                h_row_eval,
                h_col_eval,
                read_ts_row_eval,
                read_ts_col_eval,
                extended_final_ts_row_col_eval,
            ])
            .unwrap();

        // Calling Basefold batch open
        let mut polys = Vec::<Vec<F>>::with_capacity(9);

        reverse_index_bits_in_place(&mut p_p_prime);
        polys.push(p_p_prime);

        reverse_index_bits_in_place(&mut h_erow);
        polys.push(h_erow);

        reverse_index_bits_in_place(&mut h_ecol);
        polys.push(h_ecol);

        reverse_index_bits_in_place(&mut h_val);
        polys.push(h_val);

        reverse_index_bits_in_place(&mut h_row);
        polys.push(h_row);

        reverse_index_bits_in_place(&mut h_col);
        polys.push(h_col);

        reverse_index_bits_in_place(&mut read_ts_row);
        polys.push(read_ts_row);

        reverse_index_bits_in_place(&mut read_ts_col);
        polys.push(read_ts_col);

        reverse_index_bits_in_place(&mut extended_final_ts_row_col);
        polys.push(extended_final_ts_row_col);

        let random_combiners = transcript.squeeze_challenges(polys.len());

        batch_sum_check_rp.reverse();

        let mut evals = Vec::<F>::with_capacity(9);
        evals.push(p_p_prime_eval);
        evals.push(h_erow_eval);
        evals.push(h_ecol_eval);
        evals.push(h_val_eval);
        evals.push(h_row_eval);
        evals.push(h_col_eval);
        evals.push(read_ts_row_eval);
        evals.push(read_ts_col_eval);
        evals.push(extended_final_ts_row_col_eval);

        basefold_batch_open::<F, H, S>(
            &pp.basefold,
            &mut polys,
            &random_combiners,
            &batch_sum_check_rp,
            &p_p_prime_commit,
            &h_erow_ecol_commit,
            &pp.trusted_commit,
            &evals,
            transcript,
        );

        Ok(())
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        Self::Polynomial: 'a,
        Self::Commitment: 'a,
    {
        Ok(())
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<Vec<Self::Commitment>, Error> {
        let roots = transcript.read_commitments(num_polys).unwrap();

        Ok(roots
            .iter()
            .map(|r| BrakingbaseCommitment::from_root(r.clone()))
            .collect_vec())
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error> {
        let num_rows = vp.brakedown_num_rows;
        let codeword_len = vp.brakedown_codeword_len;
        let row_len = vp.brakedown_row_len;

        let (mut x_0, mut x_1) = point_to_tensor_for_commit(num_rows, point);

        let p_p_prime_commit = transcript.read_commitment().unwrap();
        // Read all the queried columns and check their Merkle paths
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        let mut col_idx = vec![0 as usize; vp.num_brakedown_queries];
        let mut cols = Vec::<F>::new();
        let mut paths = Vec::new();
        for i in 0..vp.num_brakedown_queries {
            col_idx[i] = squeeze_challenge_idx(transcript, codeword_len);
            let col = transcript.read_field_elements(vp.brakedown_num_rows)?;
            let path = transcript.read_commitments(depth)?;
            cols.extend(col);
            paths.push(path);
        }
        (0..vp.num_brakedown_queries).into_par_iter().for_each(|i| {
            let col = cols[i * vp.brakedown_num_rows..(i + 1) * vp.brakedown_num_rows].to_vec();
            let path = &paths[i];
            // verify merkle tree opening
            let mut hasher = H::new();
            let mut output = {
                for elem in col.iter() {
                    hasher.update_field_element(elem);
                }
                hasher.finalize_fixed_reset()
            };
            for (idx, neighbor) in path.iter().enumerate() {
                if ((col_idx[i] >> idx) & 1) == 0 {
                    hasher.update(&output);
                    hasher.update(neighbor);
                } else {
                    hasher.update(neighbor);
                    hasher.update(&output);
                }
                output = hasher.finalize_fixed_reset();
            }
            if &output != &comm.root {
                panic!("Invalid merkle tree opening");
            }
        });
        drop(paths);

        let mut u = transcript.squeeze_challenges(row_len.ilog2().try_into().unwrap());
        let p_prime_at_u = transcript.read_field_element()?;
        // let mut sum_check_val = F::ZERO;
        let challenges = transcript.squeeze_challenges(vp.num_brakedown_queries);
        let random_combiners = transcript.squeeze_challenges(2);
        let mut sum_check_val = (0..vp.num_brakedown_queries)
            .into_par_iter()
            .fold(
                || F::ZERO,
                |acc, j| {
                    let sum_check_val_i = (0..vp.brakedown_num_rows)
                        .into_iter()
                        .fold(F::ZERO, |acc, i| {
                            acc + x_0[i] * cols[j * vp.brakedown_num_rows + i]
                        });
                    acc + sum_check_val_i * challenges[j]
                },
            )
            .reduce_with(|acc, val| acc + val)
            .unwrap()
            * random_combiners[0]
            + p_prime_at_u * random_combiners[1];

        let sum_check_rounds = (2 * row_len).next_power_of_two().ilog2();
        let mut first_sum_check_random_points = vec![F::ZERO; sum_check_rounds as usize];
        for i in 0..sum_check_rounds as usize {
            let mut a = transcript.read_field_elements(3).unwrap();

            if sum_check_val != a[2].double() + a[1] + a[0] {
                return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
            }
            let r = transcript.squeeze_challenge();
            first_sum_check_random_points[i] = r;
            sum_check_val = a[2] + (a[1] + a[0] * r) * r;
        }

        let witness_evals = transcript.read_field_elements(3).unwrap();
        let h_eval = witness_evals[0];
        let p_eval = witness_evals[1];
        let p_prime_eval = witness_evals[2];
        let r = first_sum_check_random_points[0];
        let p_p_prime_eval = (F::ONE - r) * p_eval + r * p_prime_eval;

        /*evaluating mask at first_sum_check_random_points */

        let mask_eval = (0..vp.num_brakedown_queries)
            .into_par_iter()
            .fold(
                || F::ZERO,
                |acc, i| {
                    let val = col_idx[i] as u32;
                    let mut prod_term = challenges[i];
                    for j in 0..first_sum_check_random_points.len() {
                        if ((val << (31 - j)) >> 31) == 1 {
                            prod_term *= first_sum_check_random_points
                                [first_sum_check_random_points.len() - 1 - j];
                        } else {
                            prod_term *= F::ONE
                                - first_sum_check_random_points
                                    [first_sum_check_random_points.len() - 1 - j];
                        }
                    }
                    acc + prod_term
                },
            )
            .reduce_with(|acc, val| acc + val)
            .unwrap();

        let final_value =
            (random_combiners[0] * mask_eval + random_combiners[1] * h_eval) * p_p_prime_eval;
        if sum_check_val != final_value {
            return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
        }

        let h_erow_ecol_commit = transcript.read_commitment().unwrap();

        /*SECOND SUM_CHECK VERIFICATION */
        let mut sum_check_val = h_eval;

        let sum_check_rounds = vp.basefold_poly_size.ilog2();
        let mut second_sum_check_random_points = vec![F::ZERO; sum_check_rounds as usize];
        for i in 0..sum_check_rounds as usize {
            let mut a = transcript.read_field_elements(4).unwrap();
            if sum_check_val != a[3].double() + a[2] + a[1] + a[0] {
                return Err(Error::InvalidPcsOpen("Second Sum check failed".to_string()));
            }
            let r = transcript.squeeze_challenge();
            second_sum_check_random_points[i] = r;
            sum_check_val = a[3] + (a[2] + (a[1] + a[0] * r) * r) * r;
        }
        let h_val_eval = transcript.read_field_element().unwrap();
        let h_erow_eval = transcript.read_field_element().unwrap();
        let h_ecol_eval = transcript.read_field_element().unwrap();

        let final_value = h_val_eval * h_erow_eval * h_ecol_eval;
        if sum_check_val != final_value {
            return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
        }

        let gamma_tau = transcript.squeeze_challenges(2);

        let mut padded_u = [F::ZERO].to_vec();
        padded_u.extend(&u);

        let (expected_eval, combiners, random_points1, output_layer_eval1) =
            gkr_verifier::<F>((2 * row_len).ilog2() as usize, transcript, 4);
        let final_ts_row_eval = transcript.read_field_element().unwrap();
        let final_ts_col_eval = transcript.read_field_element().unwrap();
        input_layer_check1(
            &gamma_tau,
            &first_sum_check_random_points,
            &padded_u,
            &combiners,
            &random_points1,
            expected_eval,
            4,
            final_ts_row_eval,
            final_ts_col_eval,
        );
        let (expected_eval, combiners, random_points2, output_layer_eval2) =
            gkr_verifier::<F>(vp.basefold_poly_size.ilog2() as usize, transcript, 4);

        assert_eq!(
            output_layer_eval1[0]
                * output_layer_eval1[1]
                * output_layer_eval2[0]
                * output_layer_eval2[1],
            output_layer_eval1[2]
                * output_layer_eval1[3]
                * output_layer_eval2[2]
                * output_layer_eval2[3],
            "output layer check failed for row"
        );

        assert_eq!(
            output_layer_eval1[4]
                * output_layer_eval1[5]
                * output_layer_eval2[4]
                * output_layer_eval2[5],
            output_layer_eval1[6]
                * output_layer_eval1[7]
                * output_layer_eval2[6]
                * output_layer_eval2[7],
            "output layer check failed for col"
        );
        let h_row_eval_rp2 = transcript.read_field_element().unwrap();
        let h_col_eval_rp2 = transcript.read_field_element().unwrap();
        let read_ts_row_eval_rp2 = transcript.read_field_element().unwrap();
        let read_ts_col_eval_rp2 = transcript.read_field_element().unwrap();
        let h_erow_eval_rp2 = transcript.read_field_element().unwrap();
        let h_ecol_eval_rp2 = transcript.read_field_element().unwrap();
        let input_layer_evaluations = [
            h_erow_eval_rp2,
            h_ecol_eval_rp2,
            h_row_eval_rp2,
            h_col_eval_rp2,
            read_ts_row_eval_rp2,
            read_ts_col_eval_rp2,
        ]
        .to_vec();
        input_layer_check2(
            &gamma_tau,
            expected_eval,
            &combiners,
            4,
            &input_layer_evaluations,
        );

        //Extended random points for p+p' corresponding to first_sum_check_random_points;
        let mut p_p_prime_fsrp = vec![
            F::ZERO;
            second_sum_check_random_points.len()
                - first_sum_check_random_points.len()
        ];
        p_p_prime_fsrp.extend(&first_sum_check_random_points);

        //Extended random points for p+p' corresponding to u;
        let mut p_p_prime_rp_u = vec![F::ZERO; second_sum_check_random_points.len() - 1 - u.len()];
        p_p_prime_rp_u.push(F::ONE);
        p_p_prime_rp_u.append(&mut u);

        //Extended random points for p+p' corresponding to x0;
        let mut p_p_prime_rp_x0 =
            vec![F::ZERO; second_sum_check_random_points.len() - 1 - x_1.len().ilog2() as usize];
        p_p_prime_rp_x0.push(F::ZERO);

        let mut point_clone = point.to_vec();
        p_p_prime_rp_x0.append(&mut point_clone[(x_0.len().ilog2() as usize)..].to_vec());

        let p_p_prime_eval_fsrp = (F::ONE - first_sum_check_random_points[0]) * p_eval
            + first_sum_check_random_points[0] * p_prime_eval;
        let p_p_prime_rp_u_eval = transcript.read_field_element().unwrap();
        let p_p_prime_rp_x0_eval = transcript.read_field_element().unwrap();

        // Need to sample an extra random point to combine values here
        let r = transcript.squeeze_challenge();
        let final_ts_row_col_eval = (F::ONE - r) * final_ts_row_eval + r * final_ts_col_eval;
        let mut final_ts_row_col_rp =
            vec![F::ZERO; second_sum_check_random_points.len() - 1 - random_points1.len()];
        final_ts_row_col_rp.push(r);
        final_ts_row_col_rp.extend(random_points1);

        let mut rx = Vec::new();
        rx.push(&p_p_prime_fsrp);
        rx.push(&p_p_prime_rp_u);
        rx.push(&p_p_prime_rp_x0);
        rx.push(&second_sum_check_random_points);
        rx.push(&random_points2);
        rx.push(&final_ts_row_col_rp);

        let mut initial_claimed_evals = [
            p_p_prime_eval_fsrp,
            p_p_prime_rp_u_eval,
            p_p_prime_rp_x0_eval,
            h_val_eval,
            h_erow_eval,
            h_ecol_eval,
        ]
        .to_vec();
        initial_claimed_evals.extend(&input_layer_evaluations);
        initial_claimed_evals.push(final_ts_row_col_eval);

        let batch_sum_check_random_combiner = transcript.squeeze_challenges(13);

        let claimed_eval = initial_claimed_evals
            .iter()
            .zip(batch_sum_check_random_combiner.iter())
            .fold(F::ZERO, |acc, (val1, val2)| acc + (*val1 * *val2));
        let (evals, mut batch_sum_check_rp) = batch_sum_check_verifier::<F>(
            &rx,
            claimed_eval,
            transcript,
            &batch_sum_check_random_combiner,
        );

        let random_combiners = transcript.squeeze_challenges(evals.len());

        batch_sum_check_rp.reverse();

        // return this
        basefold_batch_verify::<F, H, S>(
            &vp.basefold,
            &random_combiners,
            &batch_sum_check_rp,
            &p_p_prime_commit,
            &h_erow_ecol_commit,
            &vp.trusted_commit,
            &evals,
            transcript,
        )
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
    ) -> Result<(), Error>
    where
        Self::Commitment: 'a,
    {
        Ok(())
    }
}

pub fn get_timestamps<F: PrimeField>(
    row: &Vec<F>,
    col: &Vec<F>,
    memory_size: usize,
    actual_reads: usize,
) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
    let num_reads = row.len();
    let mut read_ts_row = vec![F::ZERO; num_reads];
    let mut read_ts_col = vec![F::ZERO; num_reads];

    let mut final_ts_row = vec![F::ZERO; memory_size];
    let mut final_ts_col = vec![F::ZERO; memory_size];

    for i in 0..actual_reads {
        let mut bytes = [0; size_of::<u32>()];
        bytes.copy_from_slice(&row[i].to_repr().as_ref()[..size_of::<u32>()]);
        let row_idx = u32::from_le_bytes(bytes) as usize;
        read_ts_row[i] = final_ts_row[row_idx];
        final_ts_row[row_idx] += F::ONE;

        let mut bytes = [0; size_of::<u32>()];
        bytes.copy_from_slice(&col[i].to_repr().as_ref()[..size_of::<u32>()]);
        let col_idx = u32::from_le_bytes(bytes) as usize;
        read_ts_col[i] = final_ts_col[col_idx];
        final_ts_col[col_idx] += F::ONE;
    }

    // Handling dummy reads
    for i in actual_reads..num_reads {
        let mut bytes = [0; size_of::<u32>()];
        bytes.copy_from_slice(&row[0].to_repr().as_ref()[..size_of::<u32>()]);
        let row_idx = u32::from_le_bytes(bytes) as usize;
        read_ts_row[i] = final_ts_row[row_idx];
        final_ts_row[row_idx] += F::ONE;

        let mut bytes = [0; size_of::<u32>()];
        bytes.copy_from_slice(&col[0].to_repr().as_ref()[..size_of::<u32>()]);
        let col_idx = u32::from_le_bytes(bytes) as usize;
        read_ts_col[i] = final_ts_col[col_idx];
        final_ts_col[col_idx] += F::ONE;
    }

    (read_ts_row, final_ts_row, read_ts_col, final_ts_col)
}

fn squeeze_challenge_idx<F: PrimeField>(
    transcript: &mut impl FieldTranscript<F>,
    cap: usize,
) -> usize {
    let challenge = transcript.squeeze_challenge();
    let mut bytes = [0; size_of::<u32>()];
    bytes.copy_from_slice(&challenge.to_repr().as_ref()[..size_of::<u32>()]);
    (u32::from_le_bytes(bytes) as usize) % cap
}

//first_sum_check_prover(). Call the function here with p_p_prime, mask, H(X,U), the two random points, and transcript as input here.
//TODO:- Check if we can combine polynomials and run sum check.
pub fn first_sum_check_prover<F, H, S>(
    sum_check_rounds: usize,
    mut p_p_prime: Vec<F>,
    mut mask: Vec<F>,
    mut h: Vec<F>,
    random_combiners: Vec<F>,
    first_sum_check_random_points: &mut Vec<F>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let f_2_inv = F::from(2 as u64).invert().unwrap();
    for i in 0..sum_check_rounds {
        let (a1_0, a1_1, a1_2, a2_0, a2_1, a2_2) = (0..mask.len() / 2)
            .into_par_iter()
            .map(|iter| {
                let iter2 = iter + mask.len() / 2;
                let a1_0 = mask[iter] * p_p_prime[iter];
                let a1_1 = mask[iter2] * p_p_prime[iter2];
                let a1_2 = (mask[iter2].double() - mask[iter])
                    * (p_p_prime[iter2].double() - p_p_prime[iter]);
                let a2_0 = h[iter] * p_p_prime[iter];
                let a2_1 = h[iter2] * p_p_prime[iter2];
                let a2_2 =
                    (h[iter2].double() - h[iter]) * (p_p_prime[iter2].double() - p_p_prime[iter]);

                (a1_0, a1_1, a1_2, a2_0, a2_1, a2_2)
            })
            .reduce_with(
                |(acc0, acc1, acc2, acc3, acc4, acc5), (a1_0, a1_1, a1_2, a2_0, a2_1, a2_2)| {
                    (
                        acc0 + a1_0,
                        acc1 + a1_1,
                        acc2 + a1_2,
                        acc3 + a2_0,
                        acc4 + a2_1,
                        acc5 + a2_2,
                    )
                },
            )
            .unwrap();

        let a_0 = random_combiners[0] * a1_0 + random_combiners[1] * a2_0;
        let a_1 = random_combiners[0] * a1_1 + random_combiners[1] * a2_1;
        let a_2 = random_combiners[0] * a1_2 + random_combiners[1] * a2_2;
        let a_0_f_2_inv = a_0 * f_2_inv;
        let a_2_f_2_inv = a_2 * f_2_inv;
        let polynomial_current_round = [
            a_0_f_2_inv - a_1 + a_2_f_2_inv,
            -(a_0_f_2_inv.double() + a_0_f_2_inv) + a_1.double() - a_2_f_2_inv,
            a_0,
        ]
        .to_vec();
        transcript.write_field_elements(&polynomial_current_round);
        let r = transcript.squeeze_challenge();
        first_sum_check_random_points[i] = r;

        mask = par_fold_by_msb(&mask, r);
        p_p_prime = par_fold_by_msb(&p_p_prime, r);
        h = par_fold_by_msb(&h, r);
    }
    transcript.write_field_element(&h[0]);
}

//second_sum_check_prover(). Call the function here with H_val. H_erow, H_ecol, and transcript as input here.
pub fn second_sum_check_prover<F, H, S>(
    sum_check_rounds: usize,
    mut h_erow: Vec<F>,
    mut h_ecol: Vec<F>,
    mut h_val: Vec<F>,
    second_sum_check_random_points: &mut Vec<F>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let f_6 = F::from(6 as u64);
    let f_2_inv = F::from(2 as u64).invert().unwrap();
    let f_3_inv = F::from(3 as u64).invert().unwrap();
    let f_6_inv = f_6.invert().unwrap();
    for i in 0..sum_check_rounds {
        let (a_0, a_1, a_2, a_minus_one) = (0..h_erow.len() / 2)
            .into_par_iter()
            .map(|iter| {
                let iter2 = iter + h_erow.len() / 2;
                let a_0 = h_erow[iter] * h_ecol[iter] * h_val[iter];
                let a_1 = h_erow[iter2] * h_ecol[iter2] * h_val[iter2];
                let a_2 = (h_erow[iter2].double() - h_erow[iter])
                    * (h_ecol[iter2].double() - h_ecol[iter])
                    * (h_val[iter2].double() - h_val[iter]);
                let a_minus_one = (-h_erow[iter2] + h_erow[iter].double())
                    * (-h_ecol[iter2] + h_ecol[iter].double())
                    * (-h_val[iter2] + h_val[iter].double());

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
        transcript.write_field_elements(&polynomial_current_round);

        let r = transcript.squeeze_challenge();
        second_sum_check_random_points[i] = r;

        h_erow = par_fold_by_msb(&h_erow, r);
        h_ecol = par_fold_by_msb(&h_ecol, r);
        h_val = par_fold_by_msb(&h_val, r);
    }
    transcript.write_field_element(&h_val[0]);
    transcript.write_field_element(&h_erow[0]);
    transcript.write_field_element(&h_ecol[0]);
}

pub(crate) fn batch_sum_check_prover<
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
>(
    polys: &mut Vec<Vec<F>>,
    mut eqs: Vec<Vec<F>>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> (F, Vec<F>) {
    let sum_check_rounds = polys[0].len().trailing_zeros() as usize;
    // random points over the sum check rounds
    let mut random_points = vec![F::ZERO; sum_check_rounds];

    for round in 0..sum_check_rounds {
        let evals: Vec<Vec<F>> = polys
            .into_par_iter()
            .zip(&eqs)
            .map(|(poly, eq_coeff)| {
                let halfsize = poly.len() / 2;
                let mut eval = vec![F::ZERO; 3];
                (eval[0], eval[1], eval[2]) = (0..halfsize)
                    .into_par_iter()
                    .map(|k| {
                        let coeff_k = eq_coeff[k];
                        let coeff_k1 = eq_coeff[k + halfsize];
                        let poly_k = poly[k];
                        let poly_k1 = poly[k + halfsize];
                        (
                            coeff_k * poly_k,
                            coeff_k1 * poly_k1,
                            (coeff_k1.double() - coeff_k) * (poly_k1.double() - poly_k),
                        )
                    })
                    .fold_with(
                        (F::ZERO, F::ZERO, F::ZERO),
                        |(acc0, acc1, acc2), (val0, val1, val2)| {
                            (acc0 + val0, acc1 + val1, acc2 + val2)
                        },
                    )
                    .reduce_with(|(acc0, acc1, acc2), (val0, val1, val2)| {
                        (acc0 + val0, acc1 + val1, acc2 + val2)
                    })
                    .unwrap();
                eval
            })
            .collect();

        let mut combined_eval = vec![F::ZERO; 3];
        for j in 0..evals.len() {
            for i in 0..3 {
                combined_eval[i] += evals[j][i]
            }
        }

        len_3_interpolate(&mut combined_eval);
        transcript.write_field_elements(&combined_eval);

        let r_i = transcript.squeeze_challenge();
        random_points[round] = r_i;

        polys
            .iter_mut()
            .zip(eqs.iter_mut())
            .for_each(|(poly, eq_coeff)| {
                *eq_coeff = par_fold_by_msb(eq_coeff, r_i);
                *poly = par_fold_by_msb(poly, r_i);
            });
    }
    transcript.write_field_element(&eqs[0][0]);
    (eqs[0][0], random_points)
}

pub fn batch_sum_check_verifier<F: PrimeField + Serialize + DeserializeOwned>(
    r_x: &Vec<&Vec<F>>,
    claimed_eval: F,
    transcript: &mut impl FieldTranscriptRead<F>,
    batch_sum_check_random_combiner: &Vec<F>,
) -> (Vec<F>, Vec<F>) {
    let mut actual_result = claimed_eval;
    let mut r_y = Vec::new();
    for var in 0..r_x[0].len() {
        let poly = transcript.read_field_elements(3).unwrap();
        let mut previous_result = poly[0];
        for i in 0..poly.len() {
            previous_result += poly[i];
        }
        let r = transcript.squeeze_challenge();
        r_y.push(r);
        assert_eq!(actual_result, previous_result, "failed at round {}", var);
        actual_result = eval(&poly, r);
    }
    //final layer result
    let mut expected_result = F::ZERO;
    let p_p_prime_eval = transcript.read_field_element().unwrap();
    let h_val_eval = transcript.read_field_element().unwrap();
    let h_erow_eval = transcript.read_field_element().unwrap();
    let h_ecol_eval = transcript.read_field_element().unwrap();
    let h_row_eval = transcript.read_field_element().unwrap();
    let h_col_eval = transcript.read_field_element().unwrap();
    let read_ts_row_eval = transcript.read_field_element().unwrap();
    let read_ts_col_eval = transcript.read_field_element().unwrap();
    let extended_final_ts_row_col_eval = transcript.read_field_element().unwrap();

    let evals = [
        p_p_prime_eval,
        h_erow_eval,
        h_ecol_eval,
        h_val_eval,
        h_row_eval,
        h_col_eval,
        read_ts_row_eval,
        read_ts_col_eval,
        extended_final_ts_row_col_eval,
    ]
    .to_vec();

    let evaluations = [
        p_p_prime_eval,
        (batch_sum_check_random_combiner[3] * h_val_eval)
            + (batch_sum_check_random_combiner[4] * h_erow_eval)
            + (batch_sum_check_random_combiner[5] * h_ecol_eval),
        (batch_sum_check_random_combiner[6] * h_erow_eval)
            + (batch_sum_check_random_combiner[7] * h_ecol_eval)
            + (batch_sum_check_random_combiner[8] * h_row_eval)
            + (batch_sum_check_random_combiner[9] * h_col_eval)
            + (batch_sum_check_random_combiner[10] * read_ts_row_eval)
            + (batch_sum_check_random_combiner[11] * read_ts_col_eval),
        batch_sum_check_random_combiner[12] * extended_final_ts_row_col_eval,
    ]
    .to_vec();
    let first_three_combined = (0..3).fold(F::ZERO, |acc, i| {
        acc + (batch_sum_check_random_combiner[i] * evaluate_eq(&r_x[i], &r_y))
    });
    let eq_evaluations: Vec<F> = (3..r_x.len()).map(|i| evaluate_eq(&r_x[i], &r_y)).collect();
    let mut combined_eq_evaluations = [first_three_combined].to_vec();
    combined_eq_evaluations.extend(&eq_evaluations);

    let expected_result = evaluations
        .into_iter()
        .zip(combined_eq_evaluations.into_iter())
        .fold(F::ZERO, |acc, (eval, eq)| acc + (eval * eq));

    assert_eq!(
        actual_result, expected_result,
        "batch sum check final layer check failed"
    );
    (evals, r_y)
}

fn evaluate_H<F: PrimeField>(H: &ParityCheckMatrix<F>, u: &Vec<F>, size: usize) -> Vec<F> {
    let mut H_at_u = vec![F::ZERO; size];
    let tensor_u = point_to_tensor(1, u).1;
    (0..H.row.len()).for_each(|i| {
        H_at_u[H.row[i]] += H.val[i] * tensor_u[H.col[i]];
    });
    H_at_u
}

fn compute_oracle_poly<F: PrimeField>(coeffs: &Vec<usize>, point: &Vec<F>) -> Vec<F> {
    coeffs.par_iter().map(|coeff| eq(*coeff, point)).collect()
}

//TODO:-  Can be made better. Too hacky.
pub fn basefold_batch_commit<F, H, S>(
    pp: &BasefoldProverParams<F>,
    polys: &mut Vec<Vec<F>>,
) -> BasefoldBatchCommitment<F, H>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    // The input polynomials are in Type 1 representation. Saving this representation.
    let bh_evals: Vec<Type1Polynomial<F>> = polys
        .par_iter()
        .map(|poly| Type1Polynomial {
            poly: poly.to_vec(),
        })
        .collect();

    // Basefold uses Type 2 representation for encoding. Converting to this representation.

    // let mut coeffs_vec = vec![vec![F::ZERO; polys[0].len()]; polys.len()];
    let coeffs_vec: Vec<Vec<F>> = polys
        .par_iter_mut()
        .map(|poly| {
            reverse_index_bits_in_place(poly);
            let mut temp = Type2Polynomial { poly: poly.clone() };
            basefold::interpolate_over_boolean_hypercube_with_copy::<F>(&temp)
                .0
                .poly
        })
        .collect();

    let polys_type_2: Vec<Type2Polynomial<F>> = coeffs_vec
        .par_iter()
        .map(|poly| Type2Polynomial {
            poly: poly.to_vec(),
        })
        .collect();

    // Calling Basefold's encoding function.
    let codewords: Vec<Type1Polynomial<F>> = polys_type_2
        .par_iter()
        .map(|poly| {
            if (pp.rs_basecode) {
                let mut basecode = basefold::encode_rs_basecode(
                    &poly,
                    1 << pp.log_rate,
                    1 << (pp.num_vars - pp.num_rounds),
                );
                assert_eq!(basecode.poly.len() > 0, true);

                basefold::evaluate_over_foldable_domain_2(
                    pp.num_vars - pp.num_rounds + pp.log_rate,
                    pp.log_rate,
                    basecode,
                    &pp.table,
                )
            } else {
                basefold::evaluate_over_foldable_domain(pp.log_rate, poly.clone(), &pp.table)
            }
        })
        .collect();

    // Constructing a common Merkle tree for all codewords
    let codeword_tree = batch_merkelize::<F, H>(&codewords);

    BasefoldBatchCommitment {
        codewords,
        codeword_tree,
        bh_evals,
    }
}

// Can be optimised. merkle_tree[0] not required.
fn batch_merkelize<F: PrimeField, H: Hash>(vecs: &Vec<Type1Polynomial<F>>) -> Vec<Vec<Output<H>>> {
    let temp: Vec<usize> = (0..vecs[0].poly.len()).collect();
    let mut hashes: Vec<Output<H>> = (0..vecs[0].poly.len())
        .into_par_iter()
        .map(|j| {
            let mut hasher = H::new();
            (0..vecs.len()).for_each(|i| hasher.update_field_element(&(vecs[i]).poly[j]));
            hasher.finalize_fixed()
        })
        .collect();

    let mut merkle_tree = Vec::<Vec<Output<H>>>::new();
    let depth = hashes.len().ilog2();

    merkle_tree.push(hashes);
    for i in 1..=depth {
        hashes = merkle_tree[(i - 1) as usize]
            .par_chunks_exact(2)
            .map(|elems| {
                let mut hasher = H::new();
                hasher.update(&elems[0]);
                hasher.update(&elems[1]);
                hasher.finalize_fixed()
            })
            .collect();
        merkle_tree.push(hashes);
    }
    merkle_tree[1..].to_vec()
}

pub fn basefold_batch_open<F, H, S>(
    pp: &BasefoldProverParams<F>,
    polys: &mut Vec<Vec<F>>,
    random_combiners: &Vec<F>,
    point: &Vec<F>,
    comm: &BasefoldCommitment<F, H>,
    batch_comm_1: &BasefoldBatchCommitment<F, H>,
    batch_comm_2: &BasefoldBatchCommitment<F, H>,
    evals: &Vec<F>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let num_vars = pp.num_vars;
    let num_rounds = num_vars;

    assert_eq!(polys.len(), random_combiners.len());

    let mut combined_poly = vec![F::ZERO; 1 << num_vars];
    combined_poly
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, combined)| {
            for i in 0..polys.len() {
                *combined += random_combiners[i] * polys[i][j];
            }
        });

    let mut combined_poly_clone = combined_poly.clone();
    reverse_index_bits_in_place(&mut combined_poly_clone);
    let mut point_clone = point.clone();
    point_clone.reverse();

    let mut codewords = Vec::<&Vec<F>>::new();
    codewords.push(&comm.codeword.poly);
    batch_comm_1.codewords.iter().for_each(|codeword| {
        codewords.push(&codeword.poly);
    });
    batch_comm_2.codewords.iter().for_each(|codeword| {
        codewords.push(&codeword.poly);
    });

    let mut combined_codeword_0 = vec![F::ZERO; comm.codeword.poly.len()];
    combined_codeword_0
        .par_iter_mut()
        .enumerate()
        .for_each(|(j, combined)| {
            for i in 0..codewords.len() {
                *combined += random_combiners[i] * codewords[i][j]
            }
        });

    let mut combined_codeword = Type1Polynomial {
        poly: combined_codeword_0,
    };

    // Assuming all polys have len 1 << num_vars
    let mut eq_vec = vec![F::ZERO; 1 << num_vars];
    eq_vec
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, eq_element)| {
            *eq_element = eq(i, &point);
        });

    assert_eq!(eq_vec.len(), polys[0].len());

    let mut codewords = Vec::<Type1Polynomial<F>>::with_capacity(num_rounds);
    let mut merkle_trees = Vec::<Vec<Vec<Output<H>>>>::with_capacity(num_rounds);

    let f_2_inv = F::from(2 as u64).invert().unwrap();

    let mut r_point = Vec::<F>::new();

    // Commit phase

    for iter in 0..num_rounds {
        let offset = eq_vec.len() / 2;
        let (a_0, a_1, a_2) = (0..offset)
            .into_par_iter()
            .map(|i| {
                let a_0 = eq_vec[i] * combined_poly[i];
                let a_1 = eq_vec[offset + i] * combined_poly[offset + i];
                let a_2 = (eq_vec[offset + i].double() - eq_vec[i])
                    * (combined_poly[offset + i].double() - combined_poly[i]);
                (a_0, a_1, a_2)
            })
            .reduce_with(|(acc0, acc1, acc2), (a_0, a_1, a_2)| (acc0 + a_0, acc1 + a_1, acc2 + a_2))
            .unwrap();
        let a_0_f2_inv = a_0 * f_2_inv;
        let a_2_f_2_inv = a_2 * f_2_inv;
        let polynomial_current_round = [
            a_0_f2_inv - a_1 + a_2_f_2_inv,
            -(F::from(3 as u64) * a_0_f2_inv) + a_1.double() - a_2_f_2_inv,
            a_0,
        ]
        .to_vec();

        transcript.write_field_elements(&polynomial_current_round);

        let r = transcript.squeeze_challenge();
        r_point.push(r);

        eq_vec = par_fold_by_msb(&eq_vec, r);
        combined_poly = par_fold_by_msb(&combined_poly, r);

        codewords.push(basefold::basefold_one_round_by_interpolation_weights::<F>(
            &pp.table_w_weights,
            iter,
            &combined_codeword,
            r,
        ));
        combined_codeword = codewords[iter].clone();
        merkle_trees.push(basefold::merkelize::<F, H>(&combined_codeword));
        transcript.write_commitment(&merkle_trees[iter][merkle_trees[iter].len() - 1][0]);
    }

    let eq_1 = evaluate_eq(&point, &r_point);

    // Query phase
    let num_queries = pp.num_verifier_queries;

    let queries: Vec<usize> = (0..num_queries)
        .map(|_| squeeze_challenge_idx(transcript, comm.codeword.poly.len() / 2))
        .collect();

    for i in 0..queries.len() {
        let mut query = queries[i];
        transcript.write_field_element(&comm.codeword.poly[2 * query]);
        transcript.write_field_element(&comm.codeword.poly[2 * query + 1]);
        write_merkle_path::<F, H>(2 * query, &comm.codeword_tree, transcript);

        batch_comm_1.codewords.iter().for_each(|codeword| {
            transcript.write_field_element(&codeword.poly[2 * query]);
            transcript.write_field_element(&codeword.poly[2 * query + 1]);
        });

        write_merkle_path::<F, H>(2 * query, &batch_comm_1.codeword_tree, transcript);

        batch_comm_2.codewords.iter().for_each(|codeword| {
            transcript.write_field_element(&codeword.poly[2 * query]);
            transcript.write_field_element(&codeword.poly[2 * query + 1]);
        });
        write_merkle_path::<F, H>(2 * query, &batch_comm_2.codeword_tree, transcript);

        query >>= 1;

        for iter in 1..num_rounds + 1 {
            transcript.write_field_element(&codewords[iter - 1].poly[2 * query]);
            transcript.write_field_element(&codewords[iter - 1].poly[2 * query + 1]);
            write_merkle_path::<F, H>(2 * query, &merkle_trees[iter - 1], transcript);
            query >>= 1;
        }
    }
}

pub fn basefold_batch_verify<F, H, S>(
    vp: &BasefoldVerifierParams<F>,
    random_combiners: &Vec<F>,
    point: &Vec<F>,
    comm: &Output<H>,
    batch_comm_1: &Output<H>,
    batch_comm_2: &Output<H>,
    evals: &Vec<F>,
    transcript: &mut impl TranscriptRead<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F,
    >,
) -> Result<(), Error>
where
    F: PrimeField + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    let num_vars = vp.num_vars;
    let num_rounds = num_vars;
    let table_w_weights = &vp.table_w_weights;

    let f_2_inv = F::from(2 as u64).invert().unwrap();

    let mut eval = evals
        .iter()
        .zip(random_combiners)
        .fold(F::ZERO, |acc, (e, random_combiner)| {
            acc + *random_combiner * *e
        });

    // Commit phase verification
    let mut challenges = Vec::<F>::with_capacity(num_rounds);
    let mut oracles = Vec::<Output<H>>::with_capacity(num_rounds);
    for iter in 0..num_rounds {
        let a = transcript.read_field_elements(3).unwrap();
        if eval != a[2].double() + a[1] + a[0] {
            return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
        } else {
            let r = transcript.squeeze_challenge();
            eval = a[2] + (a[1] + a[0] * r) * r;
            challenges.push(r);
            let temp = transcript.read_commitment().unwrap();
            oracles.push(temp);
        }
    }

    let eq = evaluate_eq(&point, &challenges);

    let final_eval = eval * eq.invert().unwrap();
    let final_oracle = &oracles[oracles.len() - 1];
    let final_oracle_plain = vec![final_eval; 1 << vp.log_rate];
    let temp = basefold::merkelize::<F, H>(&Type1Polynomial {
        poly: final_oracle_plain,
    });
    let final_oracle_computed = &temp[temp.len() - 1][0];

    for i in 0..final_oracle.len() {
        if final_oracle_computed[i] != final_oracle[i] {
            return Err(Error::InvalidPcsOpen("Final oracle wrong".to_string()));
        }
    }

    // Verify that all oracles are correct
    let mut key: [u8; 16] = [0u8; 16];
    let mut iv: [u8; 16] = [0u8; 16];
    let mut rng = vp.rng.clone();
    rng.set_word_pos(0);
    rng.fill_bytes(&mut key);
    rng.fill_bytes(&mut iv);

    // Query phase verification
    let num_queries = vp.num_verifier_queries;
    let queries: Vec<usize> = (0..num_queries)
        .map(|i| squeeze_challenge_idx(transcript, (1 << vp.log_rate) * (1 << vp.num_vars) / 2))
        .collect();
    let mut merkle_paths1 = Vec::new();
    let mut merkle_paths2 = Vec::new();
    let mut merkle_paths3 = Vec::new();
    let mut merkle_paths4 = Vec::new();
    let mut collect_elems1 = Vec::new();
    // let mut collect_elems2 = Vec::new();
    let mut collect_hashes1 = Vec::new();
    let mut collect_hashes2 = Vec::new();
    let mut collect_elems = Vec::new();
    for i in 0..num_queries {
        let mut elems = Vec::<(F, F)>::with_capacity(num_queries);
        elems.push((F::ZERO, F::ZERO));
        // Reading the queried elements and Merkle path for p_p_prime
        let mut elem_1 = transcript.read_field_element().unwrap();
        let mut elem_2 = transcript.read_field_element().unwrap();
        elems[0].0 = random_combiners[0] * elem_1;
        elems[0].1 = random_combiners[0] * elem_2;
        let merkle_path = transcript
            .read_commitments(vp.log_rate + vp.num_vars)
            .unwrap();
        collect_elems1.push((elem_1.clone(), elem_2.clone()));
        merkle_paths1.push(merkle_path);

        // Reading the queried elements and Merkle path for h_erow and h_ecol
        let mut hasher_1 = H::new();
        let mut hasher_2 = H::new();
        for j in 0..2 {
            elem_1 = transcript.read_field_element().unwrap();
            elem_2 = transcript.read_field_element().unwrap();
            hasher_1.update_field_element(&elem_1);
            hasher_2.update_field_element(&elem_2);
            elems[0].0 += random_combiners[j + 1] * elem_1;
            elems[0].1 += random_combiners[j + 1] * elem_2;
        }

        merkle_paths2.push(
            transcript
                .read_commitments(vp.log_rate + vp.num_vars)
                .unwrap(),
        );
        collect_hashes1.push((hasher_1.finalize_fixed(), hasher_2.finalize_fixed()));

        // Reading the queried elements and Merkle path for h_val, h_row, h_col,
        // read_ts_row, read_ts_col, final_ts_row_col
        let mut hasher_1 = H::new();
        let mut hasher_2 = H::new();
        for j in 0..6 {
            elem_1 = transcript.read_field_element().unwrap();
            elem_2 = transcript.read_field_element().unwrap();
            hasher_1.update_field_element(&elem_1);
            hasher_2.update_field_element(&elem_2);
            elems[0].0 += random_combiners[j + 3] * elem_1;
            elems[0].1 += random_combiners[j + 3] * elem_2;
        }
        let merkle_path = transcript
            .read_commitments(vp.log_rate + vp.num_vars)
            .unwrap();
        merkle_paths3.push(merkle_path);
        collect_hashes2.push((hasher_1.finalize_fixed(), hasher_2.finalize_fixed()));

        let mut merkle_paths = Vec::new();
        for iter in 1..num_rounds + 1 {
            elems.push((
                transcript.read_field_element().unwrap(),
                transcript.read_field_element().unwrap(),
            ));
            merkle_paths.push(
                transcript
                    .read_commitments(vp.log_rate + vp.num_vars - iter)
                    .unwrap(),
            );
        }

        merkle_paths4.push(merkle_paths);
        collect_elems.push(elems);
    }
    (0..num_queries).into_par_iter().for_each(|i| {
        authenticate_merkle_path::<F, H>(
            2 * queries[i],
            &collect_elems1[i],
            &merkle_paths1[i],
            &comm,
        );
        authenticate_merkle_path_hash::<F, H>(
            2 * queries[i],
            &collect_hashes1[i],
            &merkle_paths2[i],
            &batch_comm_1,
        );
        authenticate_merkle_path_hash::<F, H>(
            2 * queries[i],
            &collect_hashes2[i],
            &merkle_paths3[i],
            &batch_comm_2,
        );

        (1..num_rounds + 1).into_par_iter().for_each(|iter| {
            authenticate_merkle_path::<F, H>(
                2 * queries[i] / (1 << iter),
                &collect_elems[i][iter],
                &merkle_paths4[i][iter - 1],
                &oracles[iter - 1],
            );
        });
        let elems = &collect_elems[i];
        (1..elems.len()).into_par_iter().for_each(|iter| {
            let query_idx = queries[i] / (1 << (iter - 1));
            let ri0 = reverse_bits(2 * query_idx, vp.num_vars + vp.log_rate - iter + 1);
            let ri1 = reverse_bits(2 * query_idx + 1, vp.num_vars + vp.log_rate - iter + 1);

            type Aes128Ctr64LE = ctr::Ctr32LE<aes::Aes128>;
            let mut cipher = Aes128Ctr64LE::new(
                GenericArray::from_slice(&key[..]),
                GenericArray::from_slice(&iv[..]),
            );

            let x0: F = basefold::query_point(
                1 << (num_vars + vp.log_rate - iter + 1),
                ri0,
                &mut rng.clone(),
                num_vars + vp.log_rate - iter,
                &mut cipher,
            );

            let c1 = (elems[iter - 1].0 + elems[iter - 1].1) * f_2_inv;
            let c2 = (elems[iter - 1].0 - elems[iter - 1].1) * f_2_inv * x0.invert().unwrap();
            let c = c1 + challenges[iter - 1] * c2;
            if query_idx % 2 == 0 {
                if c != elems[iter].0 {
                    panic!("ORACLES INCONSISTENT!");
                }
            } else {
                if c != elems[iter].1 {
                    panic!("ORACLES INCONSISTENT!");
                }
            }
        });
    });
    Ok(())
}

fn write_merkle_path<F: PrimeField, H: Hash>(
    mut idx: usize,
    merkle_tree: &Vec<Vec<Output<H>>>,
    transcript: &mut impl TranscriptWrite<Output<H>, F>,
) {
    let path_len = merkle_tree.len();
    idx >>= 1;
    assert!(idx < (1 << path_len - 1));
    for i in 0..path_len - 1 {
        transcript.write_commitment(&merkle_tree[i][idx ^ 1]);
        idx >>= 1;
    }
    transcript.write_commitment(&merkle_tree[path_len - 1][0]);
}

fn write_merkle_path_2<F: PrimeField, H: Hash>(
    mut idx: usize,
    merkle_tree: &Vec<Vec<Output<H>>>,
    transcript: &mut Vec<Output<H>>,
) {
    let path_len = merkle_tree.len();
    idx >>= 1;
    assert!(idx < (1 << path_len - 1));
    for i in 0..path_len - 1 {
        if idx % 2 == 0 {
            transcript.push(merkle_tree[i][idx + 1].clone());
        } else {
            transcript.push(merkle_tree[i][idx - 1].clone());
        }
        idx >>= 1;
    }
    transcript.push(merkle_tree[path_len - 1][0].clone());
}

fn authenticate_merkle_path<F: PrimeField, H: Hash>(
    mut idx: usize,
    elems: &(F, F),
    merkle_path: &Vec<Output<H>>,
    root: &Output<H>,
) -> Result<(), Error> {
    let path_len = merkle_path.len();
    idx >>= 1;
    assert!(idx < (1 << path_len - 1));
    let mut hasher = H::new();
    hasher.update_field_element(&elems.0);
    hasher.update_field_element(&elems.1);
    let mut hash = hasher.finalize_fixed();
    for i in 0..path_len - 1 {
        let mut hasher = H::new();
        if idx % 2 == 0 {
            hasher.update(&hash);
            hasher.update(&merkle_path[i]);
            hash = hasher.finalize_fixed();
        } else {
            hasher.update(&merkle_path[i]);
            hasher.update(&hash);
            hash = hasher.finalize_fixed();
        }
        idx >>= 1;
    }

    for i in 0..merkle_path[path_len - 1].len() {
        let h_1 = merkle_path[path_len - 1][i];
        let h_2 = root[i];
        let h_3 = hash[i];
        assert_eq!(h_1, h_2);
        assert_eq!(h_2, h_3);
        if h_1 != h_2 {
            panic!("ERROR in Merkle path opening!");
        }
    }
    Ok(())
}

fn authenticate_merkle_path_hash<F: PrimeField, H: Hash>(
    mut idx: usize,
    elems: &(Output<H>, Output<H>),
    merkle_path: &Vec<Output<H>>,
    root: &Output<H>,
) -> Result<(), Error> {
    let path_len = merkle_path.len();
    idx >>= 1;
    assert!(idx < (1 << path_len - 1));
    let mut hasher = H::new();
    hasher.update(&elems.0);
    hasher.update(&elems.1);
    let mut hash = hasher.finalize_fixed();
    for i in 0..path_len - 1 {
        let mut hasher = H::new();
        if idx % 2 == 0 {
            hasher.update(&hash);
            hasher.update(&merkle_path[i]);
            hash = hasher.finalize_fixed();
        } else {
            hasher.update(&merkle_path[i]);
            hasher.update(&hash);
            hash = hasher.finalize_fixed();
        }
        idx >>= 1;
    }
    for i in 0..merkle_path[path_len - 1].len() {
        let h_1 = merkle_path[path_len - 1][i];
        let h_2 = root[i];
        let h_3 = hash[i];
        assert_eq!(h_1, h_2);
        assert_eq!(h_2, h_3);
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::pcs::multilinear::basefold::Type1Polynomial;
    use crate::pcs::PolynomialCommitmentScheme;
    use crate::util::ff_255::ft127::Ft127;
    use crate::util::transcript::{
        self, FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript,
        TranscriptRead, TranscriptWrite,
    };

    use crate::{
        pcs::multilinear::{
            /*basefold::{
                basefold_one_round_by_interpolation_weights, encode_repetition_basecode,
                encode_rs_basecode, evaluate_over_foldable_domain, evaluate_over_foldable_domain_2,
                evaluate_over_foldable_domain_generic_basecode, get_table_aes,
                interpolate_over_boolean_hypercube_with_copy, log2_strict,
                multilinear_evaluation_atoz, multilinear_evaluation_ztoa, one_level_eval_hc,
                one_level_interp_hc, rand_chacha, Basefold, Type1Polynomial, Type2Polynomial,
            },*/
            test::{run_batch_commit_open_verify, run_commit_open_verify},
        },
        poly::{multilinear::MultilinearPolynomial, Polynomial},
        util::{
            ff_255::ft127,
            hash::{Hash, Keccak256, Output},
            new_fields::{Mersenne127, Mersenne61},
            play_field::PlayField,
            transcript::{Blake2sTranscript, Keccak256Transcript},
        },
    };
    //use blake2b_simd::Hash;
    use halo2_curves::{ff::Field, secp256k1::Fp};
    use plonky2_util::reverse_index_bits_in_place;
    use rand_chacha::{
        rand_core::{RngCore, SeedableRng},
        ChaCha12Rng, ChaCha8Rng,
    };
    use std::io;

    //use crate::pcs::multilinear::basefold::Instant;
    use crate::pcs::multilinear::{basefold, Basefold, BasefoldExtParams};
    use crate::util::arithmetic::PrimeField;
    use blake2::{digest::FixedOutputReset, Blake2s256};
    use halo2_curves::bn256::{Bn256, Fr};

    use crate::util::code::{
        BrakedownSpec, BrakedownSpec1, BrakedownSpec2, BrakedownSpec3, BrakedownSpec4,
        BrakedownSpec5, BrakedownSpec6, LinearCodes,
    };

    use super::{
        authenticate_merkle_path, eq, point_to_tensor, write_merkle_path, write_merkle_path_2,
        Brakingbase, BrakingbaseSpec,
    };

    #[derive(Debug)]
    pub struct Five {}

    impl BasefoldExtParams for Five {
        fn get_reps() -> usize {
            return 656;
        }

        fn get_rate() -> usize {
            return 3;
        }

        fn get_basecode_rounds() -> usize {
            return 0;
        }
        fn get_rs_basecode() -> bool {
            false // Important. Else basefold commit encodes coefficients, not evaluations.
        }
    }

    impl BrakedownSpec for Five {
        const LAMBDA: f64 = 100.0;
        const ALPHA: f64 = 0.211;
        const BETA: f64 = 0.097;
        const R: f64 = 1.616;
    }

    impl BrakingbaseSpec for Five {}

    type Pcs = Brakingbase<Ft127, Blake2s256, Five>;
    type Pcs_basefold = Basefold<Ft127, Blake2s256, Five>;

    #[test]
    fn test_merkle_paths() {
        let mut msg = vec![Fr::ZERO; 4096];
        for i in 1..msg.len() {
            msg[i] = msg[i - 1];
        }

        let merkle_tree =
            basefold::merkelize::<Fr, Blake2s256>(&Type1Polynomial { poly: msg.clone() });
        let mut transcript = Vec::<Output<Blake2s256>>::new();

        let idx = 767;
        write_merkle_path_2::<Fr, Blake2s256>(idx, &merkle_tree, &mut transcript);
        let path_len = transcript.len();

        let mut elems = (Fr::ZERO, Fr::ZERO);
        if idx % 2 == 0 {
            elems.0 = msg[idx];
            elems.1 = msg[idx + 1];
        } else {
            elems.0 = msg[idx + 1];
            elems.1 = msg[idx];
        }

        authenticate_merkle_path::<Fr, Blake2s256>(
            idx,
            &elems,
            &transcript,
            &merkle_tree[merkle_tree.len() - 1][0],
        );
    }

    #[test]
    fn test_eq() {
        let mut b = Vec::<Fr>::new();
        b.push(Fr::ONE + Fr::ONE);
        b.push(b[0] + Fr::ONE);
        b.push(b[1] + Fr::ONE);
        b.push(b[2] + Fr::ONE);
        let mut a = Fr::ZERO;
        for i in 0..16 {
            let temp = eq(i, &b);
            a += temp;
        }
    }

    #[test]
    fn test_parity_check_matrix() {
        let num_vars = 24;

        let batch_size = 1;
        let mut rng = ChaCha8Rng::from_entropy();

        let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
        let mut parity_check_matrix =
            vec![
                vec![Fr::ZERO; params.brakedown_codeword_len - params.brakedown_row_len];
                params.brakedown_codeword_len
            ];
    }

    #[test]
    fn test_point_to_tensor() {
        let mut point = [Fr::ZERO; 4];
        for i in 1..point.len() {
            point[i] = point[i - 1] + Fr::ONE;
        }
        let (x_0, x_1) = point_to_tensor(4, &point);
    }

    #[test]
    fn test_setup() {
        for num_vars in 23..25 {
            let batch_size = 1;
            let mut rng = ChaCha8Rng::from_entropy();
            let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
        }
    }

    #[test]
    fn test_trim() {
        let num_vars = 13;
        let batch_size = 1;
        let mut rng = ChaCha8Rng::from_entropy();

        let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
        let (pp, vp) = Pcs::trim(&params, 1 << num_vars, 1).unwrap();
    }

    // #[test]
    // fn test_commit() {
    //     let num_vars = 20;
    //     let batch_size = 1;
    //     let mut rng = ChaCha8Rng::from_entropy();

    //     let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
    //     let (pp, vp) = Pcs::trim(&params, 1 << num_vars, 1).unwrap();

    //     let mut rng = ChaCha8Rng::from_entropy();
    //     let poly = MultilinearPolynomial::<Fr>::new(vec![Fr::random(&mut rng); 1 << num_vars]);
    //     let comm = Pcs::commit(&pp, &poly).unwrap();
    // }
    #[test]
    fn brakingbase_commit_open_verify() {
        run_commit_open_verify::<_, Pcs, Blake2sTranscript<_>>();
    }

    // fn run_basefold_batch_open<T>() {
    //     let num_vars = 13;
    //     let batch_size = 1;
    //     let mut rng = ChaCha8Rng::from_entropy();

    //     let params = Pcs_basefold::setup(1 << num_vars, batch_size, rng).unwrap();
    //     let (pp, vp) = Pcs_basefold::trim(&params, 1 << num_vars, 1).unwrap();

    //     let mut rng = ChaCha8Rng::from_entropy();
    //     let poly_1 = MultilinearPolynomial::<Fr>::new(vec![Fr::random(&mut rng); 1 << num_vars]);
    //     let comm = Pcs_basefold::commit(&pp, &poly_1).unwrap();
    //     let poly_2 = MultilinearPolynomial::<Fr>::new(vec![Fr::random(&mut rng); 1 << num_vars]);
    //     let comm = Pcs_basefold::commit(&pp, &poly_2).unwrap();
    //     let r_1 = Fr::random(&mut rng);
    //     let r_2 = Fr::random(&mut rng);
    //     let point = vec![Fr::random(&mut rng); num_vars];
    // }

    fn vec_matrix_prod<F: PrimeField>(vc: &Vec<F>, mat: &Vec<Vec<F>>) -> Vec<F> {
        assert_eq!(vc.len(), mat.len());
        let cols = mat[0].len();
        let rows = mat.len();
        let mut res = vec![F::ZERO; cols];

        for j in 0..cols {
            for k in 0..rows {
                res[j] += vc[k] * mat[k][j];
            }
        }
        res
    }
}
