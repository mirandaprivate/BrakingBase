use crate::pcs::Commitment;
use crate::piop::sum_check::{
    classic::{ClassicSumCheck, CoefficientsProver},
    eq_xy_eval, SumCheck as _, VirtualPolynomial,
};
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
use core::fmt::Debug;
use core::hash;
use core::ptr::addr_of;
use ctr;
use ff::BatchInverter;
use generic_array::GenericArray;
use halo2_curves::bn256::{Bn256, Fr};
use rayon::iter::IntoParallelIterator;
use std::{collections::HashMap, iter, ops::Deref, time::Instant};

use plonky2_util::{log2_strict, reverse_bits, reverse_index_bits_in_place};
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha12Rng, ChaCha8Rng,
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    ParallelSlice, ParallelSliceMut,
};
use std::{borrow::Cow, marker::PhantomData, mem::size_of, slice};

use super::{BasefoldCommitment, BasefoldExtParams};
use super::MultilinearBrakedownCommitment;

const COL_SIZE: usize = 256;

#[derive(Clone, Debug)]
pub struct BrakingbaseParams<F: PrimeField, H: Hash> {
    num_vars: usize,

    brakedown: Brakedown<F>,
    brakedown_num_rows: usize,

    table_w_weights: Vec<Vec<(F, F)>>,
    table: Vec<Vec<F>>,
    num_basefold_queries: usize,
    num_basefold_rounds: Option<usize>,
    basefold_log_rate: usize,
    rs_basecode: bool,

    trusted_commits: [BasefoldCommitment<F, H> ;7],
    rng: ChaCha8Rng,
}

#[derive(Clone, Debug)]
pub struct BrakingbaseProverParams<F: PrimeField> {
    num_vars: usize,

    brakedown: Brakedown<F>,               // parity check matrix implicitly provided here
    brakedown_num_rows: usize,

    table_w_weights: Vec<Vec<(F, F)>>,
    table: Vec<Vec<F>>,
    num_basefold_queries: usize,
    num_basefold_rounds: Option<usize>,
    basefold_log_rate: usize,
    rs_basecode: bool,
}

impl<F: PrimeField> BrakingbaseProverParams<F>{
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

#[derive(Clone, Debug)]
pub struct BrakingbaseVerifierParams<F: PrimeField, H: Hash> {
    num_vars: usize,

    brakedown_num_rows: usize,
    brakedown_row_size: usize,
    num_brakedown_queries: usize,

    table_w_weights: Vec<Vec<(F, F)>>,
    num_basefold_queries: usize,
    num_basefold_rounds: usize,
    basefold_log_rate: usize,
    rs_basecode: bool,

    trusted_commits: Vec<BasefoldCommitment<F, H>>,
    rng: ChaCha8Rng,
}

pub struct BrakingbaseCommitment<F: PrimeField, H: Hash> {
    rows: Vec<F>,
    intermediate_hashes: Vec<Output<H>>,
    root: Output<H>,
}

pub trait BrakingbaseSpec: BrakedownSpec + BasefoldExtParams {
    
}

pub struct Brakingbase<F: PrimeField, H: Hash, S: BrakingbaseSpec>(PhantomData<(F, H, S)>);

impl<F, H, S> PolynomialCommitmentScheme<F> for Brakingbase<F, H, S>
where
    F: PrimeField, // + Serialize + DeserializeOwned,
    H: Hash,
    S: BrakingbaseSpec,
{
    type Param = BrakingbaseParams<F, H>;
    type ProverParam = BrakingbaseProverParams<F>;
    type VerifierParam = BrakingbaseVerifierParams<F, H>;
    type Polynomial = MultilinearPolynomial<F>;
    type Commitment = BrakingbaseCommitment<F, H>;
    type CommitmentChunk = Output<H>;

    fn setup(poly_size: usize, batch_size: usize, rng: impl RngCore) -> Result<Self::Param, Error> {
        assert!(poly_size.is_power_of_two());
        let num_vars = poly_size.ilog2() as usize;

        // Generate the Brakedown code. brakedown contains E_0 as well as H implicitly.
        let barkedown_col_size = COL_SIZE;
        let brakedown = Brakedown::new::<S>(num_vars, 20.min((1 << num_vars) - 1), COL_SIZE, rng);

        // Generate the Basefold code.  
        let log_rate = S::get_rate();
        let mut test_rng = ChaCha8Rng::from_entropy();
        let (table_w_weights, table) = super::basefold::get_table_aes(poly_size, log_rate, &mut test_rng);
        let mut rs_basecode = false;
        if S::get_rs_basecode() == true && S::get_basecode_rounds() > 0 {
            rs_basecode = true;
        }

        // Compute the trusted commits



        Ok(BrakingbaseParams {
            num_vars: num_vars,

            brakedown: brakedown,        
            brakedown_num_rows: barkedown_col_size,

            table_w_weights: table_w_weights,
            table: table,
            num_basefold_queries: S::get_reps(),
            num_basefold_rounds: Some(log2_strict(poly_size) - V::get_basecode_rounds()),
            basefold_log_rate: log_rate,
            rs_basecode: rs_basecode,

            trusted_commits: [],  //to generate using Spark

            rng: test_rng.clone(),
        })
    }

    fn commit(pp: &Self::ProverParam, poly: &Self::Polynomial) -> Result<Self::Commitment, Error> {
        validate_input("commit", pp.num_vars(), [poly], None)?;

        let row_len = pp.brakedown.row_len();

        let codeword_len = pp.brakedown.codeword_len();

        let mut rows = vec![F::ZERO; pp.brakedown_num_rows * codeword_len];

        // Encode rows. This is parallel. Do we want to make it serial for benchmarking?
	    let encoding_time = Instant::now();
        let chunk_size = div_ceil(pp.brakedown_num_rows, num_threads());

        /*rows.chunks_exact_mut(codeword_len)
            .zip(poly.evals().chunks_exact(row_len))
            .map(|row, eval| {row.copy_from_slice(eval); pp.brakedown.encode(row)});*/

        parallelize_iter(
            rows.chunks_exact_mut(chunk_size * codeword_len)
                .zip(poly.evals().chunks_exact(chunk_size * row_len)),              // All elements of row handlled together
            |(rows, evals)| {
                for (row, evals) in rows
                    .chunks_exact_mut(codeword_len)
                    .zip(evals.chunks_exact(row_len))
                {
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
            rows: rows,
            intermediate_hashes: intermediate_hashes,
            root: root
        }) 
    }

    fn open(
            pp: &Self::ProverParam,
            poly: &Self::Polynomial,
            comm: &Self::Commitment,
            point: &Point<F, Self::Polynomial>,
            eval: &F,
            transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
        ) -> Result<(), Error> {
        
    }
}
