// use crate::frontend::halo2::circuit;
use crate::pcs::multilinear::{ basefold, brakedown };
use crate::pcs::Commitment;
use crate::piop::sum_check::{ self, evaluate };
use crate::piop::sum_check::{
    classic::{ ClassicSumCheck, CoefficientsProver },
    eq_xy_eval,
    SumCheck as _,
    VirtualPolynomial,
};
use crate::util::code::{ self, ParityCheckMatrix };
use crate::{
    pcs::{
        multilinear::{ additive, validate_input },
        AdditiveCommitment,
        Evaluation,
        Point,
        PolynomialCommitmentScheme,
    },
    poly::{ multilinear::MultilinearPolynomial, Polynomial },
    util::{
        arithmetic::{ div_ceil, horner, inner_product, steps, BatchInvert, Field, PrimeField },
        code::{ Brakedown, BrakedownSpec, LinearCodes },
        expression::{ Expression, Query, Rotation },
        hash::{ Hash, Output },
        new_fields::{ Mersenne127, Mersenne61 },
        parallel::{ num_threads, parallelize, parallelize_iter },
        transcript::{ FieldTranscript, TranscriptRead, TranscriptWrite },
        BigUint,
        Deserialize,
        DeserializeOwned,
        Itertools,
        Serialize,
    },
    Error,
};
use aes::cipher::{ KeyIvInit, StreamCipher, StreamCipherSeek };
use bitvec::vec;
use halo2_proofs::poly::commitment;
use rand::random;
use core::fmt::Debug;
use core::{ hash, num, panic };
use core::ptr::addr_of;
use std::mem::swap;
use ctr;
use ff::{ derive, BatchInverter };
use generic_array::GenericArray;
use halo2_curves::bn256::{ Bn256, Fr };
use rayon::iter::IntoParallelIterator;
use std::{ collections::HashMap, iter, ops::Deref, time::Instant };

use plonky2_util::{ ceil_div_usize, log2_strict, reverse_bits, reverse_index_bits_in_place };
use rand_chacha::{ rand_core::{ RngCore, SeedableRng }, ChaCha12Rng, ChaCha8Rng };
use rayon::prelude::{
    IndexedParallelIterator,
    IntoParallelRefIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
    ParallelSlice,
    ParallelSliceMut,
};
use std::{ borrow::Cow, marker::PhantomData, mem::size_of, slice };
use super::basefold::{
    BasefoldParams,
    BasefoldProverParams,
    BasefoldVerifierParams,
    BasefoldCommitment,
    BasefoldExtParams,
    Basefold,
    Type1Polynomial,
};

use super::brakedown::{ MultilinearBrakedownCommitment };

const COL_SIZE: usize = 256;
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
    trusted_commits: Vec<BasefoldCommitment<F, H>>,
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
    trusted_commits: Vec<BasefoldCommitment<F, H>>,
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
    trusted_commits: Vec<Output<H>>, //Vec<BasefoldCommitment<F, H>>, // replace by Output<H>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingbaseCommitment<F: PrimeField, H: Hash> {
    rows: Vec<F>,
    intermediate_hashes: Vec<Output<H>>,
    root: Output<H>,
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
            root: root,
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

impl<F, H, S> PolynomialCommitmentScheme<F>
    for Brakingbase<F, H, S>
    where F: PrimeField + Serialize + DeserializeOwned, H: Hash, S: BrakingbaseSpec
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

        // Generate the Brakedown code. brakedown contains E_0 as well as H implicitly.
        let brakedown_num_rows = COL_SIZE;
        let brakedown = Brakedown::new::<S>(
            num_vars,
            (20).min((1 << num_vars) - 1),
            brakedown_num_rows,
            rng
        );
        let brakedown_row_len = brakedown.row_len();
        let brakedown_codeword_len = brakedown.codeword_len();
        let num_brakedown_queries = brakedown.num_column_opening();
        let parity_check_matrix = brakedown.parity_check_matrix();

        // Generate BaseFold parameters by running BaseFold's setup algo.
        let basefold_poly_size = (2 * parity_check_matrix.row.len()).next_power_of_two();
        let mut rng2 = ChaCha8Rng::from_entropy();
        let basefold = Basefold::<F, H, S>::setup(basefold_poly_size, batch_size, rng2).unwrap();

        // Compute the trusted commits
        let (basefold_prover_params, basefold_verifier_params) = Basefold::<F, H, S>
            ::trim(&basefold, poly_size, batch_size)
            .unwrap();
        let mut val = parity_check_matrix.clone().val;
        val.resize(basefold_poly_size, F::ZERO);
        let mut row_col = vec![F::ZERO; basefold_poly_size];

        for i in 0..parity_check_matrix.row.len() {
            row_col[i] = F::try_from(parity_check_matrix.row[i] as u64).unwrap();
        }
        let offset = basefold_poly_size / 2;
        for i in 0..parity_check_matrix.col.len() {
            row_col[offset + i] = F::try_from(parity_check_matrix.col[i] as u64).unwrap();
        }

        let (mut read_ts_row, mut final_ts_row, mut read_ts_col, mut final_ts_col) = get_timestamps(
            &row_col[0..offset],
            &row_col[offset..],
            2 * brakedown_row_len,
            parity_check_matrix.row.len()
        );

        // println!("The read_ts_row.len() is {:?}", read_ts_row.len());
        // println!("basefold_poly_size / 2 is {:?}", basefold_poly_size / 2);
        // panic!();

        // read_ts_row.resize(basefold_poly_size / 2, F::ZERO);
        // read_ts_col.resize(basefold_poly_size / 2, F::ZERO);
        final_ts_row.resize(basefold_poly_size / 2, F::ZERO);
        final_ts_col.resize(basefold_poly_size / 2, F::ZERO);
        read_ts_row.extend(final_ts_row);
        read_ts_col.extend(final_ts_col);

        let mut trusted_commits = Vec::<BasefoldCommitment<F, H>>::new();
        reverse_index_bits_in_place(&mut val); // Basefold commit accepts type 2 poly. Converts type 1 (our rep) to type 2.
        trusted_commits.push(
            Basefold::<F, H, S>
                ::commit(&basefold_prover_params, &MultilinearPolynomial::<F>::new(val))
                .unwrap()
        );
        reverse_index_bits_in_place(&mut row_col);
        trusted_commits.push(
            Basefold::<F, H, S>
                ::commit(&basefold_prover_params, &MultilinearPolynomial::<F>::new(row_col))
                .unwrap()
        );

        reverse_index_bits_in_place(&mut read_ts_row);
        trusted_commits.push(
            Basefold::<F, H, S>
                ::commit(&basefold_prover_params, &MultilinearPolynomial::<F>::new(read_ts_row))
                .unwrap()
        );
        reverse_index_bits_in_place(&mut read_ts_col);
        trusted_commits.push(
            Basefold::<F, H, S>
                ::commit(&basefold_prover_params, &MultilinearPolynomial::<F>::new(read_ts_col))
                .unwrap()
        );

        Ok(BrakingbaseParams {
            num_vars: num_vars,
            brakedown: brakedown,
            brakedown_num_rows: brakedown_num_rows,
            num_brakedown_queries, //compute
            brakedown_row_len: brakedown_row_len,
            brakedown_codeword_len: brakedown_codeword_len,
            partity_check_matrix: parity_check_matrix,
            basefold_poly_size: basefold_poly_size,
            basefold: basefold,
            basefold_prover_params: basefold_prover_params,
            basefold_verifier_params: basefold_verifier_params,
            trusted_commits: trusted_commits,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        // let (basefold_prover_params, basefold_verifier_params) = Basefold::<F, H, S>
        //     ::trim(&param.basefold, poly_size, batch_size)
        //     .unwrap();

        let mut trusted_commits = Vec::<Output<H>>::new();
        for i in 0..param.trusted_commits.len() {
            // let a = param.trusted_commits[i].codeword_tree_root().clone();
            trusted_commits.push(param.trusted_commits[i].codeword_tree_root().clone());
        }

        Ok((
            BrakingbaseProverParams {
                num_vars: param.num_vars,
                brakedown: param.brakedown.clone(),
                brakedown_num_rows: param.brakedown_num_rows,
                num_brakedown_queries: param.num_brakedown_queries,
                parity_check_matrix: param.partity_check_matrix.clone(),
                basefold_poly_size: param.basefold_poly_size,
                basefold: param.basefold_prover_params.clone(),
                trusted_commits: param.trusted_commits.clone(),
            },
            BrakingbaseVerifierParams {
                num_vars: param.num_vars,
                brakedown_num_rows: param.brakedown_num_rows,
                num_brakedown_queries: param.num_brakedown_queries,
                brakedown_row_len: param.brakedown_row_len,
                brakedown_codeword_len: param.brakedown_codeword_len,
                basefold_poly_size: param.basefold_poly_size,
                basefold: param.basefold_verifier_params.clone(),
                trusted_commits: trusted_commits,
            },
        ))
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        let row_len = pp.brakedown.row_len();

        let codeword_len = pp.brakedown.codeword_len();
        let mut rows = vec![F::ZERO; pp.brakedown_num_rows * codeword_len];

        // Encode rows. This is parallel. Do we want to make it serial for benchmarking?
        let encoding_time = Instant::now();
        let chunk_size = div_ceil(pp.brakedown_num_rows, num_threads());
        parallelize_iter(
            rows
                .chunks_mut(chunk_size * codeword_len)
                .zip(poly.evals().chunks(chunk_size * row_len)), // All elements of row handlled together
            |(rows, evals)| {
                for (row, evals) in rows.chunks_mut(codeword_len).zip(evals.chunks(row_len)) {
                    row[..evals.len()].copy_from_slice(evals);
                    pp.brakedown.encode(row);
                }
            }
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
                input.chunks(2 * chunk_size).zip(output.chunks_mut(chunk_size)),
                |(input, output)| {
                    let mut hasher = H::new();

                    for (input, output) in input.chunks_exact(2).zip(output.iter_mut()) {
                        hasher.update(&input[0]);
                        hasher.update(&input[1]);
                        hasher.finalize_into_reset(output);
                    }
                }
            );
            offset += width;
        }

        let (intermediate_hashes, root) = {
            let mut intermediate_hashes = hashes;
            let root = intermediate_hashes.pop().unwrap();
            (intermediate_hashes, root)
        };

        Ok(BrakingbaseCommitment {
            rows: rows,
            intermediate_hashes: intermediate_hashes,
            root: root,
        })
    }

    fn batch_commit<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>
    ) -> Result<Vec<Self::Commitment>, Error>
        where Self::Polynomial: 'a
    {
        let polys_vec: Vec<&Self::Polynomial> = polys
            .into_iter()
            .map(|poly| poly)
            .collect();
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
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>
    ) -> Result<(), Error> {
        let num_rows = pp.brakedown_num_rows;
        let codeword_len = pp.brakedown.codeword_len();
        let row_len = pp.brakedown.row_len();
        let basefold_poly_size = pp.basefold_poly_size;
        let (x_0, x_1) = point_to_tensor(num_rows, point);
        let mut combined_codeword = vec![F::ZERO; codeword_len];

        // Taking a linear combination of the rows of the commitment matrix
        // TODO(Vineet): Take par_iter
        for i in 0..num_rows {
            for j in 0..codeword_len {
                combined_codeword[j] += x_0[i] * comm.rows[codeword_len * i + j];
                //combined_codeword[j] += x_1[i] * comm.rows[codeword_len * i + j];
            }
        }

        // Commiting to the message and (codeword - message) parts of combined_codeword
        let mut p_p_prime: Vec<F> = Vec::new();
        let zero_padding: Vec<F> = vec![F::ZERO; 2 * row_len - codeword_len];

        // The number of coefficients in H is pp.blow_up_factor * row_len.
        for i in 0..pp.basefold_poly_size / (2 * row_len) {
            p_p_prime.extend(&combined_codeword);
            p_p_prime.extend(&zero_padding);
        }
        let p_p_prime_clone = p_p_prime.clone();
        let p_p_prime_commit = Basefold::<F, H, S>
            ::commit(&pp.basefold, &MultilinearPolynomial::new(p_p_prime_clone))
            .unwrap();
        transcript.write_commitment(p_p_prime_commit.codeword_tree_root());

        // Proximity test for the commitment matrix
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        let mut col_idx = vec![0 as usize; pp.num_brakedown_queries];
        let mut cols = vec![vec![F::ZERO; pp.brakedown_num_rows]; pp.num_brakedown_queries];
        for i in 0..pp.num_brakedown_queries {
            col_idx[i] = squeeze_challenge_idx(transcript, codeword_len);
            transcript.write_field_elements(
                comm.rows.iter().skip(col_idx[i]).step_by(codeword_len)
            )?;
            let mut col_vec = vec![F::ZERO; pp.brakedown_num_rows];
            for j in 0..pp.brakedown_num_rows {
                col_vec[j] = comm.rows[col_idx[i] + j * codeword_len];
            }
            cols[i] = col_vec;
            let mut offset = 0;
            for (idx, width) in (1..=depth)
                .rev()
                .map(|depth| 1 << depth)
                .enumerate() {
                let neighbor_idx = (col_idx[i] >> idx) ^ 1;
                transcript.write_commitment(&comm.intermediate_hashes[offset + neighbor_idx])?;
                offset += width;
            }
        }

        //TODO 1: Sample the point u.
        let mut u = transcript.squeeze_challenges(row_len.ilog2().try_into().unwrap());

        //TODO 2: Realise H(X,u) vector, that is, MLE of the matrix H with Y coordinates replaced by u. This is now a polynomial in X variables.
        let mut h = evaluate_H(&pp.parity_check_matrix, &u, pp.brakedown.codeword_len());
        h.resize(2 * row_len, F::ZERO);
        let h_clone = h.clone();
        let small_p_p_prime = p_p_prime[0..2 * row_len].to_vec();

        let p_prime_eval_u = &evaluate_poly(&small_p_p_prime[row_len..].to_vec(), &u);
        transcript.write_field_element(&p_prime_eval_u);

        let mut mask = vec![F::ZERO; 2 * row_len];
        let challenges = transcript.squeeze_challenges(pp.num_brakedown_queries);
        for i in 0..pp.num_brakedown_queries {
            mask[col_idx[i]] += challenges[i];
        }

        //TODO 4: Sample two random points here.
        let random_combiners = transcript.squeeze_challenges(2);
        // println!("random_combiners prover side: {:?}", random_combiners);

        let sum_check_rounds = (2 * row_len).next_power_of_two().ilog2() as usize;
        let mut first_sum_check_random_points = vec![F::ZERO; sum_check_rounds];

        /*TEST CODE */
        let mut sum_check_val = F::ZERO;
        for j in 0..pp.num_brakedown_queries {
            let mut sum_check_val_i = F::ZERO;
            for i in 0..pp.brakedown_num_rows {
                sum_check_val_i += x_0[i] * cols[j][i]; // make x_1[i]
            }
            sum_check_val += sum_check_val_i * challenges[j];
        }

        //prover test code:
        let mut test_val_sum_check = F::ZERO;
        for i in 0..mask.len() {
            test_val_sum_check += mask[i] * p_p_prime[i];
        }
        assert_eq!(test_val_sum_check, sum_check_val, "The first sum-check inputs are not valid");
        sum_check_val *= random_combiners[0];

        //TODO 5: Make a function called first_sum_check_prover(). Call the function here with p_p_prime, mask, H(X,U), the two random points, and transcript as input here.

        /* first_sum_check_prover() description: does the entire code of the sum_check (all the rounds) and 
        the messages are included in the transcript within the function. This function takes 
        transcript as mutable reference. 
        For reference see the sum-check implemented here https://github.com/arithmic/Dual_PCS/blob/main/Spartan/Spartan_with_gkr/src/prover/batch_eval.rs
        The len_4 interpolate here will be replaced by the expression you use at the end of  sum_check_prover_round_one or sum_check_prover_later_round
        */

        //first_sum_check_prover(sum_check_rounds, p_p_prime, mask, h, random_combiners, &mut first_sum_check_random_points, transcript);

        //prover test code:
        let mut test_val_sum_check = F::ZERO;
        for i in 0..h.len() {
            test_val_sum_check += h[i] * p_p_prime[i];
        }
        // println!("The length of h is {:?}", h.len());
        // println!("The length of p_p_prime is {:?}", p_p_prime.len());
        // println!("The test_val_sum_check is {:?}", test_val_sum_check);
        let temp = evaluate_poly(&small_p_p_prime[row_len..].to_vec(), &u);
        assert_eq!(test_val_sum_check, temp, "The first sum_check inputs are not valid");
        test_val_sum_check *= random_combiners[1];
        // println!("Combined sum_check_val prover side = {:?}", test_val_sum_check + sum_check_val);

        first_sum_check_prover::<F, H, S>(
            sum_check_rounds,
            small_p_p_prime,
            mask,
            h,
            random_combiners,
            &mut first_sum_check_random_points,
            transcript
        );

        //TODO 3: evaluate h, p, p_prime at first_sum_check_random_points. Shouldn't folding in the sum-check give this?
        let h_eval = evaluate_poly(&h_clone, &first_sum_check_random_points);
        let p_eval = partial_evaluate_poly(
            &p_p_prime[0..row_len].to_vec(),
            &first_sum_check_random_points,
            1
        ); // Suboptimal as to_vec() copies

        let p_prime_eval = partial_evaluate_poly(
            &p_p_prime[row_len..2 * row_len].to_vec(),
            &first_sum_check_random_points,
            1
        ); // Suboptimal as to_vec() copies
        transcript.write_field_elements([h_eval, p_eval, p_prime_eval].iter());

        //TODO 6.1: Commit to H_erow, H_ecol using Basefold
        //TODO 6.2(Bhargav): Compute H_val -- Check sum_check_rounds

        //TODO: Q Why are we doubling h_val. This can be basefold_size/2 right?
        let mut h_val = pp.parity_check_matrix.val.clone();
        println!("The size of h_val length before appending is {:?}", h_val.len());
        h_val.resize(basefold_poly_size, F::ZERO);
        println!("The size of h_val length after appending is {:?}", h_val.len());

        let mut h_erow_ecol = compute_oracle_poly(
            &pp.parity_check_matrix.row,
            &first_sum_check_random_points
        );
        h_erow_ecol.resize(basefold_poly_size / 2, h_erow_ecol[0]);
        let mut padded_u = [F::ZERO].to_vec();
        padded_u.extend(&u);
        h_erow_ecol.extend(compute_oracle_poly(&pp.parity_check_matrix.col, &padded_u));
        h_erow_ecol.resize(basefold_poly_size, h_erow_ecol[basefold_poly_size / 2]);

        let h_erow_ecol_commit = Basefold::<F, H, S>
            ::commit(&pp.basefold, &MultilinearPolynomial::new(h_erow_ecol.clone()))
            .unwrap();
        transcript.write_commitment(h_erow_ecol_commit.codeword_tree_root());

        assert!(h_val.len().is_power_of_two());

        let sum_check_rounds = (h_val.len() / 2).ilog2() as usize; // Changed by Bhargav
        let mut second_sum_check_random_points = vec![F::ZERO; sum_check_rounds];

        let h_val_clone = h_val.clone();
        let h_erow_ecol_clone = h_erow_ecol.clone();

        //TODO 7: Make a function called second_sum_check_prover(). Call the function here with H_erow, H_ecol, H_val, and transacript
        /* second_sum_check_prover() description: does the entire code of the sum_check (all the rounds) and 
        the messages are included in the transcript within the function. This function takes 
        transcript as mutable reference. 
        For reference see the sum-check implemented here https://github.com/arithmic/Dual_PCS/blob/main/Spartan/Spartan_with_gkr/src/prover/batch_eval.rs.
        The sum-check expression is H_erow\cdot H_ecol \cdot H_eval, and hence would need len_4 interpolate.
        */
        // let mut test_h_eval = F::ZERO;
        // for i in 0..h_val.len() / 2 {
        //     test_h_eval += h_val[i] * h_erow_ecol[i] * h_erow_ecol[basefold_poly_size / 2 + i];
        // }

        // if test_h_eval != h_eval {
        //     println!("Second sum-check input wrong on prover side");
        // }

        second_sum_check_prover::<F, H, S>(
            sum_check_rounds,
            h_erow_ecol[0..basefold_poly_size / 2].to_vec(),
            h_erow_ecol[basefold_poly_size / 2..].to_vec(),
            h_val[0..basefold_poly_size / 2].to_vec(),
            &mut second_sum_check_random_points,
            transcript
        );

        let h_val_eval = evaluate_poly(&h_val_clone, &second_sum_check_random_points);
        let h_erow_eval1 = evaluate_poly(
            &h_erow_ecol_clone[0..basefold_poly_size / 2].to_vec(),
            &second_sum_check_random_points
        );
        let h_ecol_eval1 = &evaluate_poly(
            &h_erow_ecol_clone[basefold_poly_size / 2..].to_vec(),
            &second_sum_check_random_points
        );

        transcript.write_field_element(&h_val_eval);
        transcript.write_field_element(&h_erow_eval1); // suboptimal
        transcript.write_field_element(&h_ecol_eval1);

        /* GRAND PRODUCT CHECKS */
        //TODO 8: Incorporate GKR from https://github.com/arithmic/Dual_PCS/tree/main/Grand_product/grand_product_with_gkr to our code.
        //This might need some work, and we might have to sit down with Ashish for this.
        // We could alternatively also implement Quarks.
        //Call the grand-product check argument. In total we would have 4 grand-product checks.

        //Quarks:
        //TODO 8.1: Sample two random points gamma, tau.
        let gamma_tau = transcript.squeeze_challenges(2);

        //TODO 8.2: Build 4*2 vectors
        /* polynomials required: hrow, h_erow, hrow_read_ts, hrow_final_ts, hcol, h_ecol, hcol_read_ts, hcol_final_ts */

        let mut h_row = vec![F::ZERO; pp.parity_check_matrix.row.len()];
        let mut h_col = vec![F::ZERO; pp.parity_check_matrix.col.len()];
        for i in 0..pp.parity_check_matrix.row.len() {
            h_row[i] = F::try_from(pp.parity_check_matrix.row[i] as u64).unwrap();
        }
        for i in 0..pp.parity_check_matrix.col.len() {
            h_col[i] = F::try_from(pp.parity_check_matrix.col[i] as u64).unwrap();
        }
        h_row.resize(h_row.len().next_power_of_two(), h_row[0]);
        h_col.resize(h_col.len().next_power_of_two(), h_col[0]);

        let mut read_ts_row: Vec<F> =
            pp.trusted_commits[2].bh_evals.poly[0..basefold_poly_size / 2].to_vec();
        let mut final_ts_row: Vec<F> =
            pp.trusted_commits[2].bh_evals.poly[
                basefold_poly_size / 2..basefold_poly_size / 2 + 2 * row_len
            ].to_vec();
        let mut read_ts_col: Vec<F> =
            pp.trusted_commits[3].bh_evals.poly[0..basefold_poly_size / 2].to_vec();
        let mut final_ts_col: Vec<F> =
            pp.trusted_commits[3].bh_evals.poly[
                basefold_poly_size / 2..basefold_poly_size / 2 + 2 * row_len
            ].to_vec();

        let mut circuit_1 = vec![F::ZERO; 2 * basefold_poly_size];
        let mut circuit_2 = vec![F::ZERO; 2 * basefold_poly_size];
        let mut circuit_3 = vec![F::ZERO; 2 * basefold_poly_size];
        let mut circuit_4 = vec![F::ZERO; 2 * basefold_poly_size];

        // Check range upper bounds with Vineet
        // Lots of 1s at the end. Verifier will have to take care of them.

        let mut final_ts_new = vec![F::ZERO; final_ts_row.len()];

        let mut final_ts_new = vec![F::ZERO; final_ts_row.len()];

        // Circuit 1.
        // Memory.
        let mut offset = 0;
        for i in 0..2 * row_len {
            circuit_1[i] =
                F::from_u128(i as u128) +
                gamma_tau[0] * eq(i, &first_sum_check_random_points) -
                gamma_tau[1];
        }
        // Padding memory with zeros.
        for i in 2 * row_len..basefold_poly_size / 2 {
            circuit_1[i] = F::from_u128(i as u128) - gamma_tau[1];
        }
        // Performing reads.
        offset += basefold_poly_size / 2;
        for i in 0..pp.parity_check_matrix.row.len() {
            circuit_1[offset + i] =
                h_row[i] +
                gamma_tau[0] * h_erow_ecol[i] +
                gamma_tau[0] * gamma_tau[0] * (read_ts_row[i] + F::ONE) -
                gamma_tau[1];
            // let mut bytes = [0; size_of::<u64>()];
            // bytes.copy_from_slice(&h_row[i].to_repr().as_ref()[..size_of::<u64>()]);
            // final_ts_new[(u64::from_le_bytes(bytes) as usize)] += F::ONE;
            // if i < 8 {
            //     println!("Actual: {}, {:?}, {:?}", i, read_ts_row[i], final_ts_new[(u64::from_le_bytes(bytes) as usize)]);
            // }
        }
        // Performing dummy reads of the first location in memory.
        for i in pp.parity_check_matrix.row.len()..basefold_poly_size / 2 {
            circuit_1[offset + i] =
                h_row[0] +
                gamma_tau[0] * h_erow_ecol[0] +
                gamma_tau[0] * gamma_tau[0] * (read_ts_row[i] + F::ONE) -
                gamma_tau[1];
            // let mut bytes = [0; size_of::<u32>()];
            // bytes.copy_from_slice(&h_row[0].to_repr().as_ref()[..size_of::<u32>()]);
            // final_ts_new[(u32::from_le_bytes(bytes) as usize)] += F::ONE;
            // println!("Actual: {}, {:?}", i, read_ts_row[i]);
        }
        // println!("Basefold poly size / 2 = {}", basefold_poly_size/2 );

        // Circuit 2.
        // Performing reads.
        let mut offset = 0;
        for i in 0..pp.parity_check_matrix.row.len() {
            circuit_2[i] =
                h_row[i] +
                gamma_tau[0] * h_erow_ecol[i] +
                gamma_tau[0] * gamma_tau[0] * read_ts_row[i] -
                gamma_tau[1];
        }
        // Performing dummy reads.
        for i in pp.parity_check_matrix.row.len()..basefold_poly_size / 2 {
            circuit_2[i] =
                h_row[0] +
                gamma_tau[0] * h_erow_ecol[0] +
                gamma_tau[0] * gamma_tau[0] * read_ts_row[i] -
                gamma_tau[1];
        }
        offset += basefold_poly_size / 2;
        // Final memory.
        for i in 0..2 * row_len {
            circuit_2[offset + i] =
                F::from_u128(i as u128) +
                gamma_tau[0] * eq(i, &first_sum_check_random_points) +
                gamma_tau[0] * gamma_tau[0] * final_ts_row[i] -
                gamma_tau[1];
        }
        // Padding final memory with zeros.
        for i in 2 * row_len..basefold_poly_size / 2 {
            circuit_2[offset + i] = F::from_u128(i as u128) - gamma_tau[1];
        }

        // for i in 0..final_ts_row.len() {
        //     if final_ts_row[i] != final_ts_new[i] {
        //         println!("Wrong fts at index: {}, {:?}, {:?}", i, final_ts_row[i] - final_ts_new[i], final_ts_new[i] - final_ts_row[i]);
        //     }
        // }

        // Test code.
        let mut p1 = F::ONE;
        let mut p2 = F::ONE;
        for i in 0..basefold_poly_size {
            p1 *= circuit_1[i];
            p2 *= circuit_2[i];
        }
        // println!("The cirucits should output: {:?}, {:?}", p1, p2);

        // Circuit 3.
        // Memory.
        let mut offset = 0;
        for i in 0..2 * row_len {
            circuit_3[i] = F::from_u128(i as u128) + gamma_tau[0] * eq(i, &padded_u) - gamma_tau[1];
        }
        // Padding memory with zeros.
        for i in 2 * row_len..basefold_poly_size / 2 {
            circuit_3[i] = F::from_u128(i as u128) - gamma_tau[1];
        }
        // Performing reads.
        offset += basefold_poly_size / 2;
        for i in 0..pp.parity_check_matrix.col.len() {
            circuit_3[offset + i] =
                h_col[i] +
                gamma_tau[0] * h_erow_ecol[basefold_poly_size / 2 + i] +
                gamma_tau[0] * gamma_tau[0] * (read_ts_col[i] + F::ONE) -
                gamma_tau[1];
            // let mut bytes = [0; size_of::<u64>()];
            // bytes.copy_from_slice(&h_col[i].to_repr().as_ref()[..size_of::<u64>()]);
            // final_ts_new[u64::from_le_bytes(bytes) as usize] += F::ONE;
            /*if i > pp.parity_check_matrix.row.len() - 32 {
                println!("Actual: {}, {:?}, {:?}", i, read_ts_col[i], final_ts_new[(u64::from_le_bytes(bytes) as usize)]);
            }*/
        }
        // Performing dummy reads of the first location in memory.
        for i in pp.parity_check_matrix.col.len()..basefold_poly_size / 2 {
            circuit_3[offset + i] =
            h_col[0] +
                gamma_tau[0] * h_erow_ecol[basefold_poly_size / 2] +
                gamma_tau[0] * gamma_tau[0] * (read_ts_col[i] + F::ONE) -
                gamma_tau[1];
            let mut bytes = [0; size_of::<u64>()];
            bytes.copy_from_slice(&F::ZERO.to_repr().as_ref()[..size_of::<u64>()]);
            final_ts_new[u64::from_le_bytes(bytes) as usize] += F::ONE;
            // if i > basefold_poly_size / 2 - 32 {
            //     println!("Actual: {}, {:?}, {:?}", i, read_ts_col[i], final_ts_new[(u64::from_le_bytes(bytes) as usize)]);
            // }
        }
        // println!("Basefold poly size / 2 = {}", basefold_poly_size/2 );

        // Circuit 4.
        // Performing reads.
        let mut offset = 0;
        for i in 0..pp.parity_check_matrix.col.len() {
            circuit_4[i] =
                h_col[i] +
                gamma_tau[0] * h_erow_ecol[basefold_poly_size / 2 + i] +
                gamma_tau[0] * gamma_tau[0] * read_ts_col[i] -
                gamma_tau[1];
        }
        // Performing dummy reads.
        for i in pp.parity_check_matrix.col.len()..basefold_poly_size / 2 {
            circuit_4[i] =
            h_col[0] +
                gamma_tau[0] * h_erow_ecol[basefold_poly_size / 2] + // h_erow_ecol[basefold_poly_size/2] +
                gamma_tau[0] * gamma_tau[0] * read_ts_col[i] -
                gamma_tau[1];
        }
        offset += basefold_poly_size / 2;
        // Final memory.
        for i in 0..2 * row_len {
            circuit_4[offset + i] =
                F::from_u128(i as u128) +
                gamma_tau[0] * eq(i, &padded_u) +
                gamma_tau[0] * gamma_tau[0] * final_ts_col[i] -
                gamma_tau[1];
        }
        // Padding final memory with zeros.
        for i in 2 * row_len..basefold_poly_size / 2 {
            circuit_4[offset + i] = F::from_u128(i as u128) - gamma_tau[1];
        }

        // for i in 0..final_ts_new.len() {
        //     if final_ts_new[i] != final_ts_col[i] {
        //         println!("Bad index: {}", i);
        //         println!("{:?}, {:?}", final_ts_new[i], final_ts_col[i]);
        //     }
        // }
        // Test code.
        let mut p1 = F::ONE;
        let mut p2 = F::ONE;
        for i in 0..basefold_poly_size {
            p1 *= circuit_3[i];
            p2 *= circuit_4[i];
        }
        // println!("The cirucits should output: {:?}, {:?}", p1, p2);

        create_grand_prod_circ(&mut circuit_1);
        create_grand_prod_circ(&mut circuit_2);
        create_grand_prod_circ(&mut circuit_3);
        create_grand_prod_circ(&mut circuit_4);
        // println!(
        //     "But they output: {:?}, {:?}",
        //     circuit_1[2 * basefold_poly_size - 2],
        //     circuit_2[2 * basefold_poly_size - 2]
        // );
        // println!(
        //     "But they output: {:?}, {:?}",
        //     circuit_3[2 * basefold_poly_size - 2],
        //     circuit_4[2 * basefold_poly_size - 2]
        // );

        //TODO 8.3: Commit to 4 vectors
        let circuit_11_commit = Basefold::<F, H, S>
            ::commit(
                &pp.basefold,
                &MultilinearPolynomial::new(circuit_1[basefold_poly_size..].to_vec())
            )
            .unwrap();
        let circuit_21_commit = Basefold::<F, H, S>
            ::commit(
                &pp.basefold,
                &MultilinearPolynomial::new(circuit_2[basefold_poly_size..].to_vec())
            )
            .unwrap();
        let circuit_31_commit = Basefold::<F, H, S>
            ::commit(
                &pp.basefold,
                &MultilinearPolynomial::new(circuit_3[basefold_poly_size..].to_vec())
            )
            .unwrap();
        let circuit_41_commit = Basefold::<F, H, S>
            ::commit(
                &pp.basefold,
                &MultilinearPolynomial::new(circuit_2[basefold_poly_size..].to_vec())
            )
            .unwrap();
        transcript.write_commitment(circuit_11_commit.codeword_tree_root());
        transcript.write_commitment(circuit_21_commit.codeword_tree_root());
        transcript.write_commitment(circuit_31_commit.codeword_tree_root());
        transcript.write_commitment(circuit_41_commit.codeword_tree_root());

        //TODO 8.4: Send claimed values of 4 grand-product checks
        transcript.write_field_element(&circuit_1[2 * basefold_poly_size - 2]);
        transcript.write_field_element(&circuit_2[2 * basefold_poly_size - 2]);
        transcript.write_field_element(&circuit_3[2 * basefold_poly_size - 2]);
        transcript.write_field_element(&circuit_4[2 * basefold_poly_size - 2]);
        // println!("{:?} {:?}", circuit_1[2 * basefold_poly_size - 2], circuit_2[2 * basefold_poly_size - 2]);
        // println!("{:?} {:?}", circuit_3[2 * basefold_poly_size - 2], circuit_4[2 * basefold_poly_size - 2]);

        //TODO 8.5: Sample 4 random points
        let quarks_binding_variables = transcript.squeeze_challenges(
            basefold_poly_size.ilog2() as usize
        );
        let quarks_random_combiner = transcript.squeeze_challenges(4);

        //TODO 8.6: Run 4 sum-checks in parallel for  all 4 circuits with quarks_sum_check_prover. Syntax given below.
        let sum_check_rounds = basefold_poly_size.ilog2() as usize;

        let mut quarks_sum_check_random_points = vec![F::ZERO; sum_check_rounds];

        let mut eq_random = point_to_tensor(1, &quarks_binding_variables).1; // vec![F::ZERO; circuit_1.len()/2]; // Update this.

        let circuit_10 = circuit_1[..basefold_poly_size].to_vec();
        let circuit_11 = circuit_1[basefold_poly_size..].to_vec();
        let circuit_20 = circuit_2[..basefold_poly_size].to_vec();
        let circuit_21 = circuit_2[basefold_poly_size..].to_vec();
        let circuit_30 = circuit_3[..basefold_poly_size].to_vec();
        let circuit_31 = circuit_3[basefold_poly_size..].to_vec();
        let circuit_40 = circuit_4[..basefold_poly_size].to_vec();
        let circuit_41 = circuit_4[basefold_poly_size..].to_vec();

        /*Even Odd Circuits */
        let mut circuit_1_even = vec![F::ZERO; circuit_10.len()];
        for i in 0..circuit_10.len() / 2 {
            circuit_1_even[i] = circuit_10[2 * i];
            circuit_1_even[i + circuit_10.len() / 2] = circuit_11[2 * i];
        }
        let mut circuit_1_odd = vec![F::ZERO; circuit_10.len()];
        for i in 0..circuit_10.len() / 2 {
            circuit_1_odd[i] = circuit_10[2 * i + 1];
            circuit_1_odd[i + circuit_10.len() / 2] = circuit_11[2 * i + 1];
        }

        let mut circuit_2_even = vec![F::ZERO; circuit_20.len()];
        for i in 0..circuit_20.len() / 2 {
            circuit_2_even[i] = circuit_20[2 * i];
            circuit_2_even[i + circuit_20.len() / 2] = circuit_21[2 * i];
        }
        let mut circuit_2_odd = vec![F::ZERO; circuit_20.len()];
        for i in 0..circuit_20.len() / 2 {
            circuit_2_odd[i] = circuit_20[2 * i + 1];
            circuit_2_odd[i + circuit_20.len() / 2] = circuit_21[2 * i + 1];
        }

        let mut circuit_3_even = vec![F::ZERO; circuit_30.len()];
        for i in 0..circuit_30.len() / 2 {
            circuit_3_even[i] = circuit_30[2 * i];
            circuit_3_even[i + circuit_30.len() / 2] = circuit_31[2 * i];
        }
        let mut circuit_3_odd = vec![F::ZERO; circuit_30.len()];
        for i in 0..circuit_30.len() / 2 {
            circuit_3_odd[i] = circuit_30[2 * i + 1];
            circuit_3_odd[i + circuit_30.len() / 2] = circuit_31[2 * i + 1];
        }

        let mut circuit_4_even = vec![F::ZERO; circuit_40.len()];
        for i in 0..circuit_40.len() / 2 {
            circuit_4_even[i] = circuit_40[2 * i];
            circuit_4_even[i + circuit_40.len() / 2] = circuit_41[2 * i];
        }
        let mut circuit_4_odd = vec![F::ZERO; circuit_40.len()];
        for i in 0..circuit_40.len() / 2 {
            circuit_4_odd[i] = circuit_40[2 * i + 1];
            circuit_4_odd[i + circuit_40.len() / 2] = circuit_41[2 * i + 1];
        }

        /*test code */
        let mut test_val = F::ZERO;
        for i in 0..circuit_11.len() {
            test_val += eq_random[i] * (circuit_11[i] - circuit_1_even[i] * circuit_1_odd[i]);
            if circuit_11[i] != circuit_1_even[i] * circuit_1_odd[i] {
                println!("LHS != RHS at index {}", i);
            }
        }
        assert_eq!(test_val, F::ZERO, "error in cicuit 1 computation");
        println!("The value of test_val is {:?}", test_val);
        println!(
            "The number of rounds in the quarks sum check at prover side is {sum_check_rounds}"
        );
        quarks_sum_check_prover::<F, H, S>(
            sum_check_rounds,
            eq_random,
            circuit_11.clone(),
            circuit_21.clone(),
            circuit_31.clone(),
            circuit_41.clone(),
            circuit_1_even.clone(),
            circuit_1_odd.clone(),
            circuit_2_even.clone(),
            circuit_2_odd.clone(),
            circuit_3_even.clone(),
            circuit_3_odd.clone(),
            circuit_4_even.clone(),
            circuit_4_odd.clone(),
            quarks_random_combiner,
            &mut quarks_sum_check_random_points,
            transcript
        );

        println!("QUARKS SUM CHECK PROVER RAN WITHOUT ERRORS");

        // //TODO 8.8: Evaluate the polynomials at appropriate points
        let circuit11_eval1 = evaluate_poly(&circuit_11, &quarks_sum_check_random_points);
        let circuit21_eval1 = evaluate_poly(&circuit_21, &quarks_sum_check_random_points);
        let circuit31_eval1 = evaluate_poly(&circuit_31, &quarks_sum_check_random_points);
        let circuit41_eval1 = evaluate_poly(&circuit_41, &quarks_sum_check_random_points);

        transcript.write_field_element(
            &circuit11_eval1
        );
        transcript.write_field_element(
            &circuit21_eval1
        );
        transcript.write_field_element(
            &circuit31_eval1
        );
        transcript.write_field_element(
            &circuit41_eval1
        );

        transcript.write_field_element(
            &evaluate_poly(&circuit_1_even, &quarks_sum_check_random_points)
        );
        transcript.write_field_element(
            &evaluate_poly(&circuit_2_even, &quarks_sum_check_random_points)
        );
        transcript.write_field_element(
            &evaluate_poly(&circuit_3_even, &quarks_sum_check_random_points)
        );
        transcript.write_field_element(
            &evaluate_poly(&circuit_4_even, &quarks_sum_check_random_points)
        );

        transcript.write_field_element(
            &evaluate_poly(&circuit_1_odd, &quarks_sum_check_random_points)
        );
        transcript.write_field_element(
            &evaluate_poly(&circuit_2_odd, &quarks_sum_check_random_points)
        );
        transcript.write_field_element(
            &evaluate_poly(&circuit_3_odd, &quarks_sum_check_random_points)
        );
        transcript.write_field_element(
            &evaluate_poly(&circuit_4_odd, &quarks_sum_check_random_points)
        );

        /* END OF GRAND PRODUCT CHECKS */
        let r = transcript.squeeze_challenge();

        let mut circuit_eval_point = quarks_sum_check_random_points[1..].to_vec();
        circuit_eval_point.push(r);

        //Evaluations to compute evaluations of Circuit 10 and 20.
        //Send evaluation: a) h_row, h_erow, read_ts_row, final_ts_row
        let h_row_eval = evaluate_poly(&h_row, &circuit_eval_point[1..].to_vec());
        let h_erow_eval2 = evaluate_poly(
            &h_erow_ecol[0..basefold_poly_size / 2].to_vec(),
            &circuit_eval_point[1..].to_vec()
        );
        let read_ts_row_eval = evaluate_poly(&read_ts_row, &circuit_eval_point[1..].to_vec());
        final_ts_row.resize(read_ts_row.len(), F::ZERO);
        let final_ts_row_eval = evaluate_poly(&final_ts_row, &circuit_eval_point[1..].to_vec());

        //Evaluations to compute evaluations of Circuit 10 and 20.
        //Send evaluation: a) h_col, h_ecol, read_ts_col, final_ts_col
        let h_col_eval = evaluate_poly(&h_col, &circuit_eval_point[1..].to_vec());
        let h_ecol_eval2 = evaluate_poly(
            &h_erow_ecol[basefold_poly_size / 2..].to_vec(),
            &circuit_eval_point[1..].to_vec()
        );
        let read_ts_col_eval = evaluate_poly(&read_ts_col, &circuit_eval_point[1..].to_vec());
        final_ts_col.resize(read_ts_col.len(), F::ZERO);
        let final_ts_col_eval = evaluate_poly(&final_ts_col, &circuit_eval_point[1..].to_vec());


        transcript.write_field_element(&h_row_eval);
        transcript.write_field_element(&h_erow_eval2);
        transcript.write_field_element(&read_ts_row_eval);
        transcript.write_field_element(&final_ts_row_eval);
        transcript.write_field_element(&h_col_eval);
        transcript.write_field_element(&h_ecol_eval2);
        transcript.write_field_element(&read_ts_col_eval);
        transcript.write_field_element(&final_ts_col_eval);

        let circuit11_eval2 = evaluate_poly(&circuit_11, &circuit_eval_point);
        let circuit21_eval2 = evaluate_poly(&circuit_21, &circuit_eval_point);
        let circuit31_eval2 = evaluate_poly(&circuit_31, &circuit_eval_point);
        let circuit41_eval2 = evaluate_poly(&circuit_41, &circuit_eval_point);

        transcript.write_field_element(&circuit11_eval2);
        transcript.write_field_element(&circuit21_eval2);
        transcript.write_field_element(&circuit31_eval2);
        transcript.write_field_element(&circuit41_eval2);

       

        //TODO 9: Batch Evaluate
        //TODO 9.1 List of batch evaluations: 
        // a) p_eval, p_prime_eval, h_erow_eval1, h_ecol_eval1, h_val_eval,
        // b)  h_row_eval, h_erow_eval2, read_ts_row_eval, final_ts_row_eval 
        // c) h_col_eval, h_ecol_eval2, read_ts_col_eval, final_ts_col_eval
        // c) circuit11_eval1, circuit21_eval1, circuit31_eval1, circuit41_eval1, 
        // d) circuit11_eval2, circuit21_eval2, circuit31_eval2, circuit41_eval2

        //TODO 9.2 Combine p_eval and p_prime_eval, h_erow_eval and h_ecol_eval
        let r = transcript.squeeze_challenge();
        let mut point_p_p_prime_eval1 = vec![F::ZERO; circuit_eval_point.len() -1 - first_sum_check_random_points.len()];
        point_p_p_prime_eval1.push(r);
        point_p_p_prime_eval1.append(&mut first_sum_check_random_points);
        let p_p_prime_eval_1 = (F::ONE - r) * p_eval + r* p_prime_eval;

        let mut point_p_p_prime_eval2 = vec![F::ZERO; circuit_eval_point.len() -1 - first_sum_check_random_points.len()];
        point_p_p_prime_eval2.push(F::ONE);
        point_p_p_prime_eval2.append(&mut u);

        let mut point_h_erow_ecol_eval1 = vec![r];
        point_h_erow_ecol_eval1.append(&mut second_sum_check_random_points);
        let h_erow_ecol_eval1 = (F::ONE - r) * h_erow_eval1 + r * h_ecol_eval1;

        let mut point_h_erow_ecol_eval2 = vec![r];
        point_h_erow_ecol_eval2.append(&mut circuit_eval_point[1..].to_vec()); //Same for h_val_eval, and h_row_col_eval
        let h_erow_ecol_eval1 = (F::ONE - r) * h_erow_eval2 + r * h_ecol_eval2;

        let h_row_col_eval = (F::ONE - r) * h_row_eval + r * h_col_eval;
       
        

        Ok(())
    }

    fn batch_open<'a>(
        pp: &Self::ProverParam,
        polys: impl IntoIterator<Item = &'a Self::Polynomial>,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>
    )
        -> Result<(), Error>
        where Self::Polynomial: 'a, Self::Commitment: 'a
    {
        Ok(())
    }

    fn read_commitments(
        vp: &Self::VerifierParam,
        num_polys: usize,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>
    ) -> Result<Vec<Self::Commitment>, Error> {
        let roots = transcript.read_commitments(num_polys).unwrap();

        Ok(
            roots
                .iter()
                .map(|r| BrakingbaseCommitment::from_root(r.clone()))
                .collect_vec()
        )
    }

    fn verify(
        vp: &Self::VerifierParam,
        comm: &Self::Commitment,
        point: &Point<F, Self::Polynomial>,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>
    ) -> Result<(), Error> {
        let num_rows = vp.brakedown_num_rows;
        let codeword_len = vp.brakedown_codeword_len;
        let row_len = vp.brakedown_row_len;

        let (x_0, x_1) = point_to_tensor(num_rows, point);
        //let mut combined_codeword = vec![F::ZERO; codeword_len];

        let p_p_prime_commit = transcript.read_commitment().unwrap();
        // Read all the queried columns and check their Merkle paths
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        //let sum_check_val = F::ZERO;
        let mut col_idx = vec![0 as usize; vp.num_brakedown_queries];
        let mut cols = Vec::<F>::new();

        for i in 0..vp.num_brakedown_queries {
            col_idx[i] = squeeze_challenge_idx(transcript, codeword_len);
            let col = transcript.read_field_elements(vp.brakedown_num_rows)?;
            let path = transcript.read_commitments(depth)?;

            // verify merkle tree opening
            let mut hasher = H::new();
            let mut output = {
                for elem in col.iter() {
                    hasher.update_field_element(elem);
                }

                hasher.finalize_fixed_reset()
            };
            cols.extend(col);
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
                return Err(Error::InvalidPcsOpen("Invalid merkle tree opening".to_string()));
            }
        }
        // println!("Verifier HERE 0");
        let u = transcript.squeeze_challenges(row_len.ilog2().try_into().unwrap());
        let p_prime_at_u = transcript.read_field_element()?;
        let mut sum_check_val = F::ZERO;
        let challenges = transcript.squeeze_challenges(vp.num_brakedown_queries);
        let random_combiners = transcript.squeeze_challenges(2);
        // println!("random_combiners: {:?}", random_combiners);
        for j in 0..vp.num_brakedown_queries {
            let mut sum_check_val_i = F::ZERO;
            for i in 0..vp.brakedown_num_rows {
                sum_check_val_i += x_0[i] * cols[j * vp.brakedown_num_rows + i]; // make x_1[i]
            }
            sum_check_val += sum_check_val_i * challenges[j];
        }
        sum_check_val *= random_combiners[0];
        sum_check_val += p_prime_at_u * random_combiners[1];
        // println!("sum_check_val verifier side = {:?}", sum_check_val);
        let sum_check_rounds = (2 * row_len).next_power_of_two().ilog2();
        let mut first_sum_check_random_points = vec![F::ZERO; sum_check_rounds as usize];
        for i in 0..sum_check_rounds as usize {
            let mut a = transcript.read_field_elements(3).unwrap();
            //println!("Verifier side round = {}, elems = {:?}", i, a);
            if sum_check_val != (F::ONE + F::ONE) * a[2] + a[1] + a[0] {
                println!("Error in round {i}");
                return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
            }
            let r = transcript.squeeze_challenge();
            first_sum_check_random_points[i] = r;
            sum_check_val = a[2] + a[1] * r + a[0] * r * r;
        }
        let witness_evals = transcript.read_field_elements(3).unwrap();
        let h_eval = witness_evals[0];
        let p_eval = witness_evals[1];
        let p_prime_eval = witness_evals[2];
        let r = first_sum_check_random_points[0];
        // println!("r verifier side = {:?}", r);
        let p_p_prime_eval = (F::ONE - r) * p_eval + r * p_prime_eval;

        /*evaluating mask at first_sum_check_random_points */
        let mut mask_eval = F::ZERO;

        for i in 0..vp.num_brakedown_queries {
            let val = col_idx[i] as u32;
            let mut prod_term = challenges[i];
            for j in 0..first_sum_check_random_points.len() {
                if ((val << (31 - j)) >> 31) == 1 {
                    prod_term *=
                        first_sum_check_random_points[first_sum_check_random_points.len() - 1 - j];
                } else {
                    prod_term *=
                        F::ONE -
                        first_sum_check_random_points[first_sum_check_random_points.len() - 1 - j];
                }
            }

            mask_eval += prod_term;
        }

        let final_value =
            (random_combiners[0] * mask_eval + random_combiners[1] * h_eval) * p_p_prime_eval;
        if sum_check_val != final_value {
            println!("Error in final check of first sum-check");
            return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
        }

        /*END OF FIRST SUM_CHECK VERIFICATION */
        //println!("Verifier Here");
        //transcript.write_field_elements([h_eval, p_eval, p_prime_eval].iter());

        let h_erow_ecol_commit = transcript.read_commitment().unwrap();

        /*SECOND SUM_CHECK VERIFICATION */
        let mut sum_check_val = h_eval;
        //TODO (Bhargav): Passes modulo the sum_check_rounds. Needs to be determined. The expression does not hold for number of vars >13.
        let sum_check_rounds = (vp.basefold_poly_size / 2).ilog2();
        let mut second_sum_check_random_points = vec![F::ZERO; sum_check_rounds as usize];
        for i in 0..sum_check_rounds as usize {
            let mut a = transcript.read_field_elements(4).unwrap();
            if sum_check_val != (F::ONE + F::ONE) * a[3] + a[2] + a[1] + a[0] {
                println!("Error in round {i}");
                return Err(Error::InvalidPcsOpen("Second Sum check failed".to_string()));
            }
            let r = transcript.squeeze_challenge();
            second_sum_check_random_points[i] = r;
            sum_check_val = a[3] + a[2] * r + a[1] * r * r + a[0] * r * r * r;
        }
        let h_val_eval = transcript.read_field_element().unwrap();
        let h_erow_eval1 = transcript.read_field_element().unwrap();
        let h_ecol_eval1 = transcript.read_field_element().unwrap();

        // println!(
        //     "Second sum-check random point verifier side is {:?}",
        //     second_sum_check_random_points[second_sum_check_random_points.len() - 1]
        // );

        let final_value = h_val_eval * h_erow_eval1 * h_ecol_eval1;
        if sum_check_val != final_value {
            println!("Error in final check of second sum-check");
            return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
        }
        println!("2-ND SUM-CHECK DONE");
        /*END OF SECOND SUM_CHECK VERIFICATION */

        /*QUARKS SUM_CHECK VERIFICATION */
        println!("STARTING QUARKS SUM CHECK VERIFICATION");
        let gamma_tau = transcript.squeeze_challenges(2);

        let circuit_11_commit = transcript.read_commitment().unwrap();
        let circuit_21_commit = transcript.read_commitment().unwrap();
        let circuit_31_commit = transcript.read_commitment().unwrap();
        let circuit_41_commit = transcript.read_commitment().unwrap();

        let circuit_1_value = transcript.read_field_element();
        let circuit_2_value = transcript.read_field_element();
        let circuit_3_value = transcript.read_field_element();
        let circuit_4_value = transcript.read_field_element();

        assert_eq!(circuit_1_value, circuit_2_value, "grand_product check not satisfied for rows");
        assert_eq!(circuit_3_value, circuit_4_value, "grand_product check not satisfied for cols");

        let quarks_binding_variables = transcript.squeeze_challenges(
            vp.basefold_poly_size.ilog2() as usize
        );
        let quarks_random_combiner = transcript.squeeze_challenges(4);

        let mut sum_check_val = F::ZERO;
        let sum_check_rounds = vp.basefold_poly_size.ilog2();
        println!(
            "The number of rounds in the quarks sum check at verifier side is {sum_check_rounds}"
        );
        let mut quarks_sum_check_random_points = vec![F::ZERO; sum_check_rounds as usize];
        for i in 0..sum_check_rounds as usize {
            let mut a = transcript.read_field_elements(4).unwrap();
            if sum_check_val != (F::ONE + F::ONE) * a[3] + a[2] + a[1] + a[0] {
                println!("Error in round {i}");
                return Err(Error::InvalidPcsOpen("Quarks Sum check failed".to_string()));
            }
            let r = transcript.squeeze_challenge();
            if i == 0 {
                println!("The random value at round 0 of the quarks sum check is {:?}", r);
            }
            quarks_sum_check_random_points[i] = r;
            sum_check_val = a[3] + a[2] * r + a[1] * r * r + a[0] * r * r * r;
        }

        //Reading values
        let circuit11_eval = transcript.read_field_element().unwrap();
        let circuit21_eval = transcript.read_field_element().unwrap();
        let circuit31_eval = transcript.read_field_element().unwrap();
        let circuit41_eval = transcript.read_field_element().unwrap();

        let circuit1_even_eval = transcript.read_field_element().unwrap();
        let circuit2_even_eval = transcript.read_field_element().unwrap();
        let circuit3_even_eval = transcript.read_field_element().unwrap();
        let circuit4_even_eval = transcript.read_field_element().unwrap();

        let circuit1_odd_eval = transcript.read_field_element().unwrap();
        let circuit2_odd_eval = transcript.read_field_element().unwrap();
        let circuit3_odd_eval = transcript.read_field_element().unwrap();
        let circuit4_odd_eval = transcript.read_field_element().unwrap();

        //compute eq_random
        let mut eq_random_value = F::ONE;
        for i in 0..quarks_binding_variables.len() {
            eq_random_value *=
                (F::ONE - quarks_binding_variables[i]) *
                    (F::ONE - quarks_sum_check_random_points[i]) +
                quarks_binding_variables[i] * quarks_sum_check_random_points[i];
        }

        let mut final_value =
            quarks_random_combiner[0] * (circuit11_eval - circuit1_even_eval * circuit1_odd_eval);
        final_value +=
            quarks_random_combiner[1] * (circuit21_eval - circuit2_even_eval * circuit2_odd_eval);
        final_value +=
            quarks_random_combiner[2] * (circuit31_eval - circuit3_even_eval * circuit3_odd_eval);
        final_value +=
            quarks_random_combiner[3] * (circuit41_eval - circuit4_even_eval * circuit4_odd_eval);

        final_value *= eq_random_value;

        if sum_check_val != final_value {
            println!("Error in final check of quarks sum-check");
            return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
        }

        /*END OF QUARKS SUM_CHECK VERIFICATION */
        let r = transcript.squeeze_challenge();
        let mut circuit_eval_point = quarks_sum_check_random_points[1..].to_vec();
        circuit_eval_point.push(r);
       

        let h_row_eval = transcript.read_field_element().unwrap();
        let h_erow_eval2 = transcript.read_field_element().unwrap();
        let read_ts_row_eval = transcript.read_field_element().unwrap();
        let final_ts_row_eval = transcript.read_field_element().unwrap();
        let h_col_eval = transcript.read_field_element().unwrap();
        let h_ecol_eval2 = transcript.read_field_element().unwrap();
        let read_ts_col_eval = transcript.read_field_element().unwrap();
        let final_ts_col_eval = transcript.read_field_element().unwrap();

        /*Computing Circuit10 and Circuit20 evaluations */
        let mut row_idx_value = F::ZERO;
        for i in 0..circuit_eval_point.len() - 1 {
            row_idx_value +=
                F::from_u128(1 << i) * circuit_eval_point[circuit_eval_point.len() - 1 - i];
        }
        let mut extend_first_sum_check_random_points = vec![F::ZERO;circuit_eval_point.len()-1];
        let extend_length = circuit_eval_point.len() - 1 - first_sum_check_random_points.len();
        for i in 0..circuit_eval_point.len() - 1 {
            if i >= extend_length {
                extend_first_sum_check_random_points[i] = first_sum_check_random_points[
                    i - extend_length
                ];
            }
        }
        let val = eq_eval_random(
            &extend_first_sum_check_random_points,
            &circuit_eval_point[1..].to_vec()
        );

        let value1 = row_idx_value + gamma_tau[0] * val - gamma_tau[1];
        let value2 =
            h_row_eval +
            gamma_tau[0] * h_erow_eval2 +
            gamma_tau[0] * gamma_tau[0] * (read_ts_row_eval + F::ONE) -
            gamma_tau[1];
        let circuit10_eval2 =
            (F::ONE - circuit_eval_point[0]) * value1 + circuit_eval_point[0] * value2;
       

        let value2 =
            row_idx_value +
            gamma_tau[0] * val +
            gamma_tau[0] * gamma_tau[0] * final_ts_row_eval -
            gamma_tau[1];
        let value1 =
            h_row_eval +
            gamma_tau[0] * h_erow_eval2 +
            gamma_tau[0] * gamma_tau[0] * read_ts_row_eval -
            gamma_tau[1];
        let circuit20_eval2 =
            (F::ONE - circuit_eval_point[0]) * value1 + circuit_eval_point[0] * value2;
        
        /*End Of Computing Circuit10 and Circuit20 evaluations */

        /*Computing Circuit30 and Circuit40 evaluations */
        let mut extend_u = vec![F::ZERO;circuit_eval_point.len()-1];
        let extend_length = circuit_eval_point.len() - 1 - u.len();
        for i in 0..circuit_eval_point.len() - 1 {
            if i >= extend_length {
                extend_u[i] = u[
                    i - extend_length
                ];
            }
        }
        let val = eq_eval_random(
            &extend_u,
            &circuit_eval_point[1..].to_vec()
        );

        let value1 = row_idx_value + gamma_tau[0] * val - gamma_tau[1];
        let value2 =
            h_col_eval +
            gamma_tau[0] * h_ecol_eval2 +
            gamma_tau[0] * gamma_tau[0] * (read_ts_col_eval + F::ONE) -
            gamma_tau[1];
        let circuit30_eval2 =
            (F::ONE - circuit_eval_point[0]) * value1 + circuit_eval_point[0] * value2;
        //assert_eq!(circuit30_eval, test_value, "Circuit1 test values not matching");

        let value2 =
            row_idx_value +
            gamma_tau[0] * val +
            gamma_tau[0] * gamma_tau[0] * final_ts_col_eval -
            gamma_tau[1];
        let value1 =
            h_col_eval +
            gamma_tau[0] * h_ecol_eval2 +
            gamma_tau[0] * gamma_tau[0] * read_ts_col_eval -
            gamma_tau[1];
        let circuit40_eval2 =
            (F::ONE - circuit_eval_point[0]) * value1 + circuit_eval_point[0] * value2;
       
        /*End Of Computing Circuit30 and Circuit40 evaluations */

       

        let circuit11_eval2 = transcript.read_field_element().unwrap();
        let circuit21_eval2 = transcript.read_field_element().unwrap();
        let circuit31_eval2 = transcript.read_field_element().unwrap();
        let circuit41_eval2 = transcript.read_field_element().unwrap();

        //Verification of odd and even evals
        let quarks_sum_check_msb = quarks_sum_check_random_points[0];
        let circuit1_eval2 =
            (F::ONE - quarks_sum_check_msb) * circuit10_eval2 +
            quarks_sum_check_msb * circuit11_eval2;
        let circuit2_eval2 =
            (F::ONE - quarks_sum_check_msb) * circuit20_eval2 +
            quarks_sum_check_msb * circuit21_eval2;
        let circuit3_eval2 =
            (F::ONE - quarks_sum_check_msb) * circuit30_eval2 +
            quarks_sum_check_msb * circuit31_eval2;
        let circuit4_eval2 =
            (F::ONE - quarks_sum_check_msb) * circuit40_eval2 +
            quarks_sum_check_msb * circuit41_eval2;

        let circuit1_eval1 = (F::ONE - r) * circuit1_even_eval + r * circuit1_odd_eval;
        let circuit2_eval1 = (F::ONE - r) * circuit2_even_eval + r * circuit2_odd_eval;
        let circuit3_eval1 = (F::ONE - r) * circuit3_even_eval + r * circuit3_odd_eval;
        let circuit4_eval1 = (F::ONE - r) * circuit4_even_eval + r * circuit4_odd_eval;

        assert_eq!(circuit1_eval1, circuit1_eval2, "circuit values not matching");
        assert_eq!(circuit2_eval1, circuit2_eval2, "circuit values not matching");
        assert_eq!(circuit3_eval1, circuit3_eval2, "circuit values not matching");
        assert_eq!(circuit4_eval1, circuit4_eval2, "circuit values not matching");
        
        
        
        
        println!("Verifier here");

        Ok(())
    }

    fn batch_verify<'a>(
        vp: &Self::VerifierParam,
        comms: impl IntoIterator<Item = &'a Self::Commitment>,
        points: &[Point<F, Self::Polynomial>],
        evals: &[Evaluation<F>],
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>
    ) -> Result<(), Error>
        where Self::Commitment: 'a
    {
        Ok(())
    }
}

pub fn get_timestamps<F: PrimeField>(
    row: &[F],
    col: &[F],
    memory_size: usize,
    actual_reads: usize
) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
    let num_reads = row.len();
    println!("Num reads = {}", num_reads);
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
        // if i > num_reads - 32 {
        //     println!("{}, {:?}, {:?}", i, read_ts_col[i], final_ts_col[col_idx]);
        // }
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
        // if i > num_reads - 32 {
        //     println!("{}, {:?}, {:?}", i, read_ts_col[i], final_ts_col[col_idx]);
        // }
    }

    // println!("TS = {:?}", final_ts_row[0..4].to_vec());
    (read_ts_row, final_ts_row, read_ts_col, final_ts_col)
}

fn point_to_tensor<F: PrimeField>(num_rows: usize, point: &[F]) -> (Vec<F>, Vec<F>) {
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(point.len() - (num_rows.ilog2() as usize));
    let t_0 = eq_xy(lo); // switch t_0 and t_1
    let t_1 = eq_xy(hi);
    (t_0, t_1)
}

fn eq_eval_random<F: PrimeField>(random_point1: &[F], random_point2: &[F]) -> F {
    //compute eq_random
    assert_eq!(
        random_point1.len(),
        random_point2.len(),
        "The lengths of the  random points are not equal!"
    );
    let mut eq_random_value = F::ONE;
    for i in 0..random_point1.len() {
        eq_random_value *=
            (F::ONE - random_point1[i]) * (F::ONE - random_point2[i]) +
            random_point1[i] * random_point2[i];
    }
    eq_random_value
}

fn eq_xy<F: PrimeField>(y: &[F]) -> Vec<F> {
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
                next_evals.chunks_mut(chunk_size).zip(evals.chunks(chunk_size >> 1)),
                |(next_evals, evals)| expand_serial(next_evals, evals, y_i)
            );
        }
        evals = next_evals;
    }

    evals
}

fn squeeze_challenge_idx<F: PrimeField>(
    transcript: &mut impl FieldTranscript<F>,
    cap: usize
) -> usize {
    let challenge = transcript.squeeze_challenge();
    let mut bytes = [0; size_of::<u32>()];
    bytes.copy_from_slice(&challenge.to_repr().as_ref()[..size_of::<u32>()]);
    (u32::from_le_bytes(bytes) as usize) % cap
}

//first_sum_check_prover(). Call the function here with p_p_prime, mask, H(X,U), the two random points, and transcript as input here.
pub fn first_sum_check_prover<F, H, S>(
    sum_check_rounds: usize,
    mut p_p_prime: Vec<F>,
    mut mask: Vec<F>,
    mut h: Vec<F>,
    random_combiners: Vec<F>,
    first_sum_check_random_points: &mut Vec<F>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F
    >
)
    where F: PrimeField + Serialize + DeserializeOwned, H: Hash, S: BrakingbaseSpec
{
    // prover test code:
    // let mut test_val_sum_check = F::ZERO;
    // for i in 0..h.len() {
    //     test_val_sum_check += h[i] * p_p_prime[i];
    // }
    // println!("The length of h is {:?}", h.len());
    // println!("The length of p_p_prime is {:?}", p_p_prime.len());
    // println!("The test_val_sum_check is {:?}", test_val_sum_check);
    // let temp = evaluate_poly(p_p_prime[p_p_prime.len()/2..], point)
    // assert_eq!(test_val_sum_check, F::ZERO, "The first sum_check inputs are not valid");
    //transcript.write_field_elements(&sum_check_prover_round_one(&mask, &p_p_prime));
    let f_2 = F::ONE + F::ONE;
    let f_2_inv = f_2.invert().unwrap();
    let f_3 = f_2 + F::ONE;
    for i in 0..sum_check_rounds {
        let mut a1_0 = F::ZERO;
        let mut a1_1 = F::ZERO;
        let mut a1_2 = F::ZERO;
        // println!("The length of mask is {:?}", mask.len());
        for iter in 0..mask.len() / 2 {
            a1_0 += mask[iter] * p_p_prime[iter];
            a1_1 += mask[iter + mask.len() / 2] * p_p_prime[iter + mask.len() / 2];
            a1_2 +=
                (f_2 * mask[iter + mask.len() / 2] - mask[iter]) *
                (f_2 * p_p_prime[iter + mask.len() / 2] - p_p_prime[iter]);
        }

        let mut a2_0 = F::ZERO;
        let mut a2_1 = F::ZERO;
        let mut a2_2 = F::ZERO;

        for iter in 0..mask.len() / 2 {
            a2_0 += h[iter] * p_p_prime[iter];
            a2_1 += h[iter + mask.len() / 2] * p_p_prime[iter + mask.len() / 2];
            a2_2 +=
                (f_2 * h[iter + mask.len() / 2] - h[iter]) *
                (f_2 * p_p_prime[iter + mask.len() / 2] - p_p_prime[iter]);
        }

        // let mask_at_zero: F = mask[0..mask.len() / 2].iter().sum();
        // let mask_at_one: F = mask[mask.len() / 2..].iter().sum();
        // let mask_at_two = f_2 * mask_at_one - mask_at_zero;

        // let p_p_prime_at_zero: F = p_p_prime[0..p_p_prime.len() / 2].iter().sum();
        // let p_p_prime_at_one: F = p_p_prime[p_p_prime.len() / 2..].iter().sum();
        // let p_p_prime_at_two = f_2 * p_p_prime_at_one - p_p_prime_at_zero;

        // let h_at_zero: F = h[0..h.len() / 2].iter().sum();
        // let h_at_one: F = h[h.len() / 2..].iter().sum();
        // let h_at_two = f_2 * h_at_one - h_at_zero;

        let a_0 = random_combiners[0] * a1_0 + random_combiners[1] * a2_0;
        let a_1 = random_combiners[0] * a1_1 + random_combiners[1] * a2_1;
        let a_2 = random_combiners[0] * a1_2 + random_combiners[1] * a2_2;
        // let a_0 = h_at_zero * p_p_prime_at_zero;
        // let a_1 = h_at_one * p_p_prime_at_one;
        // let a_2 = h_at_two * p_p_prime_at_two;
        let polynomial_current_round = [
            a_0 * f_2_inv - a_1 + a_2 * f_2_inv,
            -(f_3 * a_0 * f_2_inv) + f_2 * a_1 - a_2 * f_2_inv,
            a_0,
        ].to_vec();
        //println!("round = {}, elems = {:?}", i, polynomial_current_round);
        transcript.write_field_elements(&polynomial_current_round);
        let r = transcript.squeeze_challenge();
        if i == 0 {
            // println!("r in the first round prover side is {:?}", r);
        }
        first_sum_check_random_points[i] = r;

        let mask_len = mask.len();
        for i in 0..mask_len / 2 {
            mask[i] = (F::ONE - r) * mask[i] + r * mask[i + mask_len / 2];
            p_p_prime[i] = (F::ONE - r) * p_p_prime[i] + r * p_p_prime[i + mask_len / 2];
            h[i] = (F::ONE - r) * h[i] + r * h[i + mask_len / 2];
        }

        mask.resize(mask_len / 2, F::ZERO);
        p_p_prime.resize(mask_len / 2, F::ZERO);
        h.resize(mask_len / 2, F::ZERO);
    }
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
        F
    >
)
    where F: PrimeField + Serialize + DeserializeOwned, H: Hash, S: BrakingbaseSpec
{
    //transcript.write_field_elements(&sum_check_prover_round_one(&mask, &p_p_prime));
    let f_2 = F::ONE + F::ONE;
    let f_2_inv = f_2.invert().unwrap();
    let f_3 = f_2 + F::ONE;
    for i in 0..sum_check_rounds {
        let mut a_0 = F::ZERO;
        let mut a_1 = F::ZERO;
        let mut a_2 = F::ZERO;
        let mut a_minus_one = F::ZERO;
        for iter in 0..h_erow.len() / 2 {
            a_0 += h_erow[iter] * h_ecol[iter] * h_val[iter];
            a_1 +=
                h_erow[iter + h_erow.len() / 2] *
                h_ecol[iter + h_erow.len() / 2] *
                h_val[iter + h_erow.len() / 2];
            a_2 +=
                (f_2 * h_erow[iter + h_erow.len() / 2] - h_erow[iter]) *
                (f_2 * h_ecol[iter + h_erow.len() / 2] - h_ecol[iter]) *
                (f_2 * h_val[iter + h_erow.len() / 2] - h_val[iter]);

            a_minus_one +=
                (-h_erow[iter + h_erow.len() / 2] + f_2 * h_erow[iter]) *
                (-h_ecol[iter + h_erow.len() / 2] + f_2 * h_ecol[iter]) *
                (-h_val[iter + h_erow.len() / 2] + f_2 * h_val[iter]);
        }

        //TODO (Bhargav): edit the following expression to derive the 4 coefficients
        let f_2 = F::ONE + F::ONE;
        let f_2_inv = f_2.invert().unwrap();
        let f_3 = f_2 + F::ONE;
        let f_3_inv = f_3.invert().unwrap();
        let f_6 = f_3 + f_3;
        let f_6_inv = f_6.invert().unwrap();
        let polynomial_current_round = [
            a_0 * f_2_inv - a_1 * f_2_inv + a_2 * f_6_inv - a_minus_one * f_6_inv,
            -a_0 + a_1 * f_2_inv + a_minus_one * f_2_inv,
            -a_0 * f_2_inv + a_1 - a_2 * f_6_inv - a_minus_one * f_3_inv,
            a_0,
        ].to_vec();
        transcript.write_field_elements(&polynomial_current_round);
        let r = transcript.squeeze_challenge();
        second_sum_check_random_points[i] = r;

        for i in 0..h_erow.len() / 2 {
            h_erow[i] = (F::ONE - r) * h_erow[i] + r * h_erow[i + h_erow.len() / 2];
            h_ecol[i] = (F::ONE - r) * h_ecol[i] + r * h_ecol[i + h_ecol.len() / 2];
            h_val[i] = (F::ONE - r) * h_val[i] + r * h_val[i + h_val.len() / 2];
        }
        h_erow.resize(h_erow.len() / 2, F::ZERO);
        h_ecol.resize(h_ecol.len() / 2, F::ZERO);
        h_val.resize(h_val.len() / 2, F::ZERO);
    }
}

pub fn quarks_sum_check_prover<F, H, S>(
    sum_check_rounds: usize,
    mut eq_random: Vec<F>,
    mut circuit_11: Vec<F>,
    mut circuit_21: Vec<F>,
    mut circuit_31: Vec<F>,
    mut circuit_41: Vec<F>,
    mut circuit_1_even: Vec<F>,
    mut circuit_1_odd: Vec<F>,
    mut circuit_2_even: Vec<F>,
    mut circuit_2_odd: Vec<F>,
    mut circuit_3_even: Vec<F>,
    mut circuit_3_odd: Vec<F>,
    mut circuit_4_even: Vec<F>,
    mut circuit_4_odd: Vec<F>,
    quarks_random_combiner: Vec<F>,
    quarks_sum_check_random_points: &mut Vec<F>,
    transcript: &mut impl TranscriptWrite<
        <Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk,
        F
    >
)
    where F: PrimeField + Serialize + DeserializeOwned, H: Hash, S: BrakingbaseSpec
{
    //transcript.write_field_elements(&sum_check_prover_round_one(&mask, &p_p_prime));
    for i in 0..sum_check_rounds {
        let f_2 = F::ONE + F::ONE;
        let f_2_inv = f_2.invert().unwrap();
        let f_3 = f_2 + F::ONE;

        let mut circuit1_at_zero = F::ZERO;
        let mut circuit1_at_one = F::ZERO;
        let mut circuit1_at_two = F::ZERO;
        let mut circuit1_at_minus_one = F::ZERO;

        let mut circuit2_at_zero = F::ZERO;
        let mut circuit2_at_one = F::ZERO;
        let mut circuit2_at_two = F::ZERO;
        let mut circuit2_at_minus_one = F::ZERO;

        let mut circuit3_at_zero = F::ZERO;
        let mut circuit3_at_one = F::ZERO;
        let mut circuit3_at_two = F::ZERO;
        let mut circuit3_at_minus_one = F::ZERO;

        let mut circuit4_at_zero = F::ZERO;
        let mut circuit4_at_one = F::ZERO;
        let mut circuit4_at_two = F::ZERO;
        let mut circuit4_at_minus_one = F::ZERO;

        for iter in 0..circuit_11.len() / 2 {
            ////Circuit computations at zero
            circuit1_at_zero +=
                eq_random[iter] * (circuit_11[iter] - circuit_1_even[iter] * circuit_1_odd[iter]);
            circuit2_at_zero +=
                eq_random[iter] * (circuit_21[iter] - circuit_2_even[iter] * circuit_2_odd[iter]);
            circuit3_at_zero +=
                eq_random[iter] * (circuit_31[iter] - circuit_3_even[iter] * circuit_3_odd[iter]);
            circuit4_at_zero +=
                eq_random[iter] * (circuit_41[iter] - circuit_4_even[iter] * circuit_4_odd[iter]);

            ////Circuit computations at one
            circuit1_at_one +=
                eq_random[iter + circuit_11.len() / 2] *
                (circuit_11[iter + circuit_11.len() / 2] -
                    circuit_1_even[iter + circuit_11.len() / 2] *
                        circuit_1_odd[iter + circuit_11.len() / 2]);
            circuit2_at_one +=
                eq_random[iter + circuit_11.len() / 2] *
                (circuit_21[iter + circuit_11.len() / 2] -
                    circuit_2_even[iter + circuit_11.len() / 2] *
                        circuit_2_odd[iter + circuit_11.len() / 2]);
            circuit3_at_one +=
                eq_random[iter + circuit_11.len() / 2] *
                (circuit_31[iter + circuit_11.len() / 2] -
                    circuit_3_even[iter + circuit_11.len() / 2] *
                        circuit_3_odd[iter + circuit_11.len() / 2]);
            circuit4_at_one +=
                eq_random[iter + circuit_11.len() / 2] *
                (circuit_41[iter + circuit_11.len() / 2] -
                    circuit_4_even[iter + circuit_11.len() / 2] *
                        circuit_4_odd[iter + circuit_11.len() / 2]);

            ////Circuit computations at two

            let val4 = f_2 * eq_random[iter + circuit_11.len() / 2] - eq_random[iter];

            let val1 = f_2 * circuit_11[iter + circuit_11.len() / 2] - circuit_11[iter];
            let val2 = f_2 * circuit_1_even[iter + circuit_11.len() / 2] - circuit_1_even[iter];
            let val3 = f_2 * circuit_1_odd[iter + circuit_11.len() / 2] - circuit_1_odd[iter];

            circuit1_at_two += val4 * (val1 - val2 * val3);

            let val1 = f_2 * circuit_21[iter + circuit_11.len() / 2] - circuit_21[iter];
            let val2 = f_2 * circuit_2_even[iter + circuit_11.len() / 2] - circuit_2_even[iter];
            let val3 = f_2 * circuit_2_odd[iter + circuit_11.len() / 2] - circuit_2_odd[iter];
            circuit2_at_two += val4 * (val1 - val2 * val3);

            let val1 = f_2 * circuit_31[iter + circuit_11.len() / 2] - circuit_31[iter];
            let val2 = f_2 * circuit_3_even[iter + circuit_11.len() / 2] - circuit_3_even[iter];
            let val3 = f_2 * circuit_3_odd[iter + circuit_11.len() / 2] - circuit_3_odd[iter];
            circuit3_at_two += val4 * (val1 - val2 * val3);

            let val1 = f_2 * circuit_41[iter + circuit_11.len() / 2] - circuit_41[iter];
            let val2 = f_2 * circuit_4_even[iter + circuit_11.len() / 2] - circuit_4_even[iter];
            let val3 = f_2 * circuit_4_odd[iter + circuit_11.len() / 2] - circuit_4_odd[iter];
            circuit4_at_two += val4 * (val1 - val2 * val3);

            ////Circuit computations at minus_one

            let val4 = -eq_random[iter + circuit_11.len() / 2] + f_2 * eq_random[iter];

            let val1 = -circuit_11[iter + circuit_11.len() / 2] + f_2 * circuit_11[iter];
            let val2 = -circuit_1_even[iter + circuit_11.len() / 2] + f_2 * circuit_1_even[iter];
            let val3 = -circuit_1_odd[iter + circuit_11.len() / 2] + f_2 * circuit_1_odd[iter];
            circuit1_at_minus_one += val4 * (val1 - val2 * val3);

            let val1 = -circuit_21[iter + circuit_11.len() / 2] + f_2 * circuit_21[iter];
            let val2 = -circuit_2_even[iter + circuit_11.len() / 2] + f_2 * circuit_2_even[iter];
            let val3 = -circuit_2_odd[iter + circuit_11.len() / 2] + f_2 * circuit_2_odd[iter];
            circuit2_at_minus_one += val4 * (val1 - val2 * val3);

            let val1 = -circuit_31[iter + circuit_11.len() / 2] + f_2 * circuit_31[iter];
            let val2 = -circuit_3_even[iter + circuit_11.len() / 2] + f_2 * circuit_3_even[iter];
            let val3 = -circuit_3_odd[iter + circuit_11.len() / 2] + f_2 * circuit_3_odd[iter];
            circuit3_at_minus_one += val4 * (val1 - val2 * val3);

            let val1 = -circuit_41[iter + circuit_11.len() / 2] + f_2 * circuit_41[iter];
            let val2 = -circuit_4_even[iter + circuit_11.len() / 2] + f_2 * circuit_4_even[iter];
            let val3 = -circuit_4_odd[iter + circuit_11.len() / 2] + f_2 * circuit_4_odd[iter];
            circuit4_at_minus_one += val4 * (val1 - val2 * val3);
        }

        let a_0 =
            quarks_random_combiner[0] * circuit1_at_zero +
            quarks_random_combiner[1] * circuit2_at_zero +
            quarks_random_combiner[2] * circuit3_at_zero +
            quarks_random_combiner[3] * circuit4_at_zero;

        let a_1 =
            quarks_random_combiner[0] * circuit1_at_one +
            quarks_random_combiner[1] * circuit2_at_one +
            quarks_random_combiner[2] * circuit3_at_one +
            quarks_random_combiner[3] * circuit4_at_one;

        let a_2 =
            quarks_random_combiner[0] * circuit1_at_two +
            quarks_random_combiner[1] * circuit2_at_two +
            quarks_random_combiner[2] * circuit3_at_two +
            quarks_random_combiner[3] * circuit4_at_two;

        let a_minus_one =
            quarks_random_combiner[0] * circuit1_at_minus_one +
            quarks_random_combiner[1] * circuit2_at_minus_one +
            quarks_random_combiner[2] * circuit3_at_minus_one +
            quarks_random_combiner[3] * circuit4_at_minus_one;

        let f_2 = F::ONE + F::ONE;
        let f_2_inv = f_2.invert().unwrap();
        let f_3 = f_2 + F::ONE;
        let f_3_inv = f_3.invert().unwrap();
        let f_6 = f_3 + f_3;
        let f_6_inv = f_6.invert().unwrap();
        let polynomial_current_round = [
            a_0 * f_2_inv - a_1 * f_2_inv + a_2 * f_6_inv - a_minus_one * f_6_inv,
            -a_0 + a_1 * f_2_inv + a_minus_one * f_2_inv,
            -a_0 * f_2_inv + a_1 - a_2 * f_6_inv - a_minus_one * f_3_inv,
            a_0,
        ].to_vec();
        transcript.write_field_elements(&polynomial_current_round);
        // println!("round = {}, points")
        let r = transcript.squeeze_challenge();

        quarks_sum_check_random_points[i] = r;

        for j in 0..circuit_11.len() / 2 {
            circuit_11[j] = (F::ONE - r) * circuit_11[j] + r * circuit_11[j + circuit_11.len() / 2];
            circuit_1_even[j] =
                (F::ONE - r) * circuit_1_even[j] + r * circuit_1_even[j + circuit_1_even.len() / 2];
            circuit_1_odd[j] =
                (F::ONE - r) * circuit_1_odd[j] + r * circuit_1_odd[j + circuit_1_odd.len() / 2];

            circuit_21[j] = (F::ONE - r) * circuit_21[j] + r * circuit_21[j + circuit_21.len() / 2];
            circuit_2_even[j] =
                (F::ONE - r) * circuit_2_even[j] + r * circuit_2_even[j + circuit_2_even.len() / 2];
            circuit_2_odd[j] =
                (F::ONE - r) * circuit_2_odd[j] + r * circuit_2_odd[j + circuit_2_odd.len() / 2];

            circuit_31[j] = (F::ONE - r) * circuit_31[j] + r * circuit_31[j + circuit_31.len() / 2];
            circuit_3_even[j] =
                (F::ONE - r) * circuit_3_even[j] + r * circuit_3_even[j + circuit_3_even.len() / 2];
            circuit_3_odd[j] =
                (F::ONE - r) * circuit_3_odd[j] + r * circuit_3_odd[j + circuit_3_odd.len() / 2];

            circuit_41[j] = (F::ONE - r) * circuit_41[j] + r * circuit_41[j + circuit_41.len() / 2];
            circuit_4_even[j] =
                (F::ONE - r) * circuit_4_even[j] + r * circuit_4_even[j + circuit_4_even.len() / 2];
            circuit_4_odd[j] =
                (F::ONE - r) * circuit_4_odd[j] + r * circuit_4_odd[j + circuit_4_odd.len() / 2];

            eq_random[j] = (F::ONE - r) * eq_random[j] + r * eq_random[j + eq_random.len() / 2];
        }
        circuit_11.resize(circuit_11.len() / 2, F::ZERO);
        circuit_1_even.resize(circuit_1_even.len() / 2, F::ZERO);
        circuit_1_odd.resize(circuit_1_odd.len() / 2, F::ZERO);

        circuit_21.resize(circuit_21.len() / 2, F::ZERO);
        circuit_2_even.resize(circuit_2_even.len() / 2, F::ZERO);
        circuit_2_odd.resize(circuit_2_odd.len() / 2, F::ZERO);

        circuit_31.resize(circuit_31.len() / 2, F::ZERO);
        circuit_3_even.resize(circuit_3_even.len() / 2, F::ZERO);
        circuit_3_odd.resize(circuit_3_odd.len() / 2, F::ZERO);

        circuit_41.resize(circuit_41.len() / 2, F::ZERO);
        circuit_4_even.resize(circuit_4_even.len() / 2, F::ZERO);
        circuit_4_odd.resize(circuit_4_odd.len() / 2, F::ZERO);

        eq_random.resize(eq_random.len() / 2, F::ZERO);
    }
}

fn evaluate_H<F: PrimeField>(H: &ParityCheckMatrix<F>, u: &Vec<F>, size: usize) -> Vec<F> {
    let mut H_at_u = vec![F::ZERO; size];
    let tensor_u = point_to_tensor(1, u).1;
    // println!("The length of tensor u {:?}", tensor_u.len());
    for i in 0..H.row.len() {
        H_at_u[H.row[i]] += H.val[i] * tensor_u[H.col[i]];
    }
    H_at_u
}

fn evaluate_poly<F: PrimeField>(coeffs: &Vec<F>, point: &Vec<F>) -> F {
    let mut eval = F::ZERO;
    let tensor_point = point_to_tensor(1, point).1;
    for i in 0..tensor_point.len() {
        eval += coeffs[i] * tensor_point[i];
    }
    eval
}

fn partial_evaluate_poly<F: PrimeField>(coeffs: &Vec<F>, point: &Vec<F>, skip: usize) -> F {
    let mut eval = F::ZERO;
    let tensor_point = point_to_tensor(1 << (point.len() - skip), point).0;
    for i in 0..tensor_point.len() {
        eval += coeffs[i] * tensor_point[i];
    }
    eval
}

// fn compute_eq_poly<F: PrimeField>(point: &Vec<F>) -> Vec<F> {
//     let mut oracle_poly = vec![F::ONE; 1];
//     for i in 0..coeffs.len() {
//         assert!(coeffs[i] < 1 << (point.len() + 1));
//         oracle_poly[i] = eq(coeffs[i], point);
//     }
//     oracle_poly
// }

fn compute_oracle_poly<F: PrimeField>(coeffs: &Vec<usize>, point: &Vec<F>) -> Vec<F> {
    let mut oracle_poly = vec![F::ZERO; coeffs.len()];
    for i in 0..coeffs.len() {
        assert!(coeffs[i] < 1 << (point.len() + 1));
        oracle_poly[i] = eq(coeffs[i], point);
    }
    oracle_poly
}

fn eq<F: PrimeField>(mut idx: usize, point: &Vec<F>) -> F {
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

fn create_grand_prod_circ<F: PrimeField>(circuit: &mut Vec<F>) {
    assert!(circuit.len().is_power_of_two());
    let mut offset_1 = circuit.len() / 2;
    let mut offset_2 = 0;
    let mut layer_size = circuit.len() / 4;
    while layer_size >= 1 {
        for i in 0..layer_size {
            circuit[offset_1 + i] = circuit[offset_2 + 2 * i] * circuit[offset_2 + 2 * i + 1];
        }
        offset_2 = offset_1;
        offset_1 += layer_size;
        layer_size /= 2;
    }
}

#[cfg(test)]
mod test {
    use crate::pcs::multilinear::brakingbase::COL_SIZE;
    use crate::pcs::PolynomialCommitmentScheme;
    use crate::util::transcript::{
        FieldTranscript,
        FieldTranscriptRead,
        FieldTranscriptWrite,
        InMemoryTranscript,
        TranscriptRead,
        TranscriptWrite,
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
            test::{ run_batch_commit_open_verify, run_commit_open_verify },
        },
        poly::{ multilinear::MultilinearPolynomial, Polynomial },
        util::{
            hash::{ Hash, Keccak256, Output },
            new_fields::{ Mersenne127, Mersenne61 },
            play_field::PlayField,
            transcript::{ Blake2sTranscript, Keccak256Transcript },
        },
    };
    use halo2_curves::{ ff::Field, secp256k1::Fp };
    use plonky2_util::reverse_index_bits_in_place;
    use rand_chacha::{ rand_core::{ RngCore, SeedableRng }, ChaCha12Rng, ChaCha8Rng };
    use std::io;

    //use crate::pcs::multilinear::basefold::Instant;
    use crate::pcs::multilinear::BasefoldExtParams;
    use crate::util::arithmetic::PrimeField;
    use blake2::{ digest::FixedOutputReset, Blake2s256 };
    use halo2_curves::bn256::{ Bn256, Fr };

    use crate::util::code::{
        BrakedownSpec,
        BrakedownSpec1,
        BrakedownSpec2,
        BrakedownSpec3,
        BrakedownSpec4,
        BrakedownSpec5,
        BrakedownSpec6,
        LinearCodes,
    };

    use super::{ create_grand_prod_circ, point_to_tensor, Brakingbase, BrakingbaseSpec };

    #[derive(Debug)]
    pub struct Five {}

    impl BasefoldExtParams for Five {
        fn get_reps() -> usize {
            return 33;
        }

        fn get_rate() -> usize {
            return 3;
        }

        fn get_basecode_rounds() -> usize {
            return 0;
        }
        fn get_rs_basecode() -> bool {
            false
        }
    }

    impl BrakedownSpec for Five {
        const LAMBDA: f64 = 100.0;
        const ALPHA: f64 = 0.211;
        const BETA: f64 = 0.097;
        const R: f64 = 1.616;
    }

    impl BrakingbaseSpec for Five {}

    type Pcs = Brakingbase<Fr, Blake2s256, Five>;

    #[test]
    fn test_create_grand_product_circuit() {
        let mut circuit = vec![Fr::ONE; 16];
        for i in 1..8 {
            circuit[i] = circuit[i - 1] + Fr::ONE;
        }
        create_grand_prod_circ(&mut circuit);
        println!("{:?}", circuit);
    }

    #[test]
    fn test_parity_check_matrix() {
        let num_vars = 16;

        let batch_size = 1;
        let mut rng = ChaCha8Rng::from_entropy();

        let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
        let mut parity_check_matrix =
            vec![vec![Fr::ZERO; params.brakedown_codeword_len - params.brakedown_row_len]; 
            params.brakedown_codeword_len];

        println!(
            "parity check matrix sparsity = {}, row len = {}",
            params.partity_check_matrix.row.len(),
            (1 << num_vars) / COL_SIZE
        );

        // for i in 0..params.partity_check_matrix.row.len() {
        //     let row = params.partity_check_matrix.row[i];
        //     let col = params.partity_check_matrix.col[i];
        //     parity_check_matrix[row][col] = params.partity_check_matrix.val[i];
        // }

        // let mut rng = ChaCha8Rng::from_entropy();
        // let mut msg = vec![Fr::random(&mut rng); params.brakedown_row_len];
        // msg.extend(vec![Fr::ZERO; params.brakedown_codeword_len - params.brakedown_row_len]);
        // params.brakedown.encode(&mut msg);
        // let mut res = vec_matrix_prod(&msg, &parity_check_matrix);

        // for i in 0..res.len() {
        //     assert_eq!(Fr::ZERO, res[i]);
        // }
        //println!("{:?}", parity_check_matrix);
    }

    #[test]
    fn test_point_to_tensor() {
        let mut point = [Fr::ZERO; 4];
        for i in 1..point.len() {
            point[i] = point[i - 1] + Fr::ONE;
        }
        let (x_0, x_1) = point_to_tensor(4, &point);
        println!("x_0 = {:?}", x_0);
        println!("{:?}, {:?}", -x_0[1], -x_0[2]);
        println!("x_1 = {:?}", x_1);
    }

    #[test]
    fn test_setup() {
        let num_vars = 15;
        let batch_size = 1;
        let mut rng = ChaCha8Rng::from_entropy();

        let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();

        println!(
            "{}, {}, {}, {}",
            params.num_vars,
            params.brakedown_row_len,
            params.brakedown_num_rows,
            params.brakedown_codeword_len
        );
        println!("{}, {}", params.basefold_poly_size, params.basefold.num_vars);
        println!("{}", params.trusted_commits.len());
        //println!("{:?}", params.basefold);
    }

    #[test]
    fn test_trim() {
        let num_vars = 13;
        let batch_size = 1;
        let mut rng = ChaCha8Rng::from_entropy();

        let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
        let (pp, vp) = Pcs::trim(&params, 1 << num_vars, 1).unwrap();
        println!(
            "{}, {}, {}, {}",
            pp.num_vars,
            pp.brakedown.row_len(),
            pp.brakedown_num_rows,
            pp.num_brakedown_queries
        );
        //println!("{:?}", params.brakedown);
        println!(
            "{}, {}, {}, {}",
            vp.num_vars,
            vp.brakedown_row_len,
            vp.brakedown_codeword_len,
            vp.brakedown_num_rows
        );
    }

    #[test]
    fn test_commit() {
        let num_vars = 13;
        let batch_size = 1;
        let mut rng = ChaCha8Rng::from_entropy();

        let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
        let (pp, vp) = Pcs::trim(&params, 1 << num_vars, 1).unwrap();

        let mut rng = ChaCha8Rng::from_entropy();
        let poly = MultilinearPolynomial::<Fr>::new(vec![Fr::random(&mut rng); 1 << num_vars]);
        let comm = Pcs::commit(&pp, &poly).unwrap();
        //println!("{:?}", poly.evals());
        println!("{}", comm.rows.len());
        //println!("{:?}", comm.rows);
    }

    #[test]
    fn test_open() {
        run_commit_open_verify::<_, Pcs, Blake2sTranscript<_>>();
    }

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
