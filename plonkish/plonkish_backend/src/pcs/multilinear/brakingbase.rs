use crate::pcs::multilinear::{basefold, brakedown};
use crate::pcs::Commitment;
use crate::piop::sum_check::{
    classic::{ClassicSumCheck, CoefficientsProver},
    eq_xy_eval, SumCheck as _, VirtualPolynomial,
};
use crate::util::code;
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
use halo2_proofs::poly::commitment;
use core::fmt::Debug;
use core::{hash, num};
use core::ptr::addr_of;
use ctr;
use ff::{derive, BatchInverter};
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
use super::basefold::{BasefoldParams, BasefoldProverParams, BasefoldVerifierParams, BasefoldCommitment, 
                        BasefoldExtParams, Basefold};
use super::brakedown::MultilinearBrakedownCommitment;

const COL_SIZE: usize = 256;
const BLOW_UP_FACTOR: usize = 16;

#[derive(Clone, Debug)]
pub struct BrakingbaseParams<F: PrimeField, H: Hash> {
    num_vars: usize,
    brakedown: Brakedown<F>,
    brakedown_num_rows: usize,
    num_brakedown_queries: usize,
    brakedown_row_len: usize,
    brakedown_codeword_len: usize,
    basefold: BasefoldParams<F>,
    trusted_commits: Vec<BasefoldCommitment<F, H>>
}

#[derive(Clone, Debug)]
pub struct BrakingbaseProverParams<F: PrimeField> {
    num_vars: usize,
    brakedown: Brakedown<F>,               // parity check matrix implicitly provided here
    brakedown_num_rows: usize,
    num_brakedown_queries: usize,
    basefold: BasefoldProverParams<F>
}

#[derive(Clone, Debug)]
pub struct BrakingbaseVerifierParams<F: PrimeField, H: Hash> {
    num_vars: usize,
    brakedown_num_rows: usize,
    num_brakedown_queries: usize,
    brakedown_row_len: usize,
    brakedown_codeword_len: usize,
    basefold: BasefoldVerifierParams<F>,
    trusted_commits: Vec<BasefoldCommitment<F, H>>
}

pub struct BrakingbaseCommitment<F: PrimeField, H: Hash> {
    rows: Vec<F>,
    intermediate_hashes: Vec<Output<H>>,
    root: Output<H>,
}

impl<F: PrimeField> BrakingbaseProverParams<F>{
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

pub trait BrakingbaseSpec: BrakedownSpec + BasefoldExtParams {
    
}

pub struct Brakingbase<F: PrimeField, H: Hash, S: BrakingbaseSpec>(PhantomData<(F, H, S)>);

impl<F, H, S> PolynomialCommitmentScheme<F> for Brakingbase<F, H, S>
where
    F: PrimeField + Serialize + DeserializeOwned,
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
        let brakedown_num_rows = COL_SIZE;
        let brakedown = Brakedown::new::<S>(num_vars, 20.min((1 << num_vars) - 1), brakedown_num_rows, rng);
        let brakedown_row_len = brakedown.row_len();
        let brakedown_codeword_len = brakedown.codeword_len();

        // Generate BaseFold parameters by running BaseFold's setup algo.
        let basefold = Basefold::<F, H, S>::setup(BLOW_UP_FACTOR*poly_size/COL_SIZE, batch_size, rng).unwrap();

        // Compute the trusted commits



        Ok(BrakingbaseParams {
            num_vars: num_vars,
            brakedown: brakedown,        
            brakedown_num_rows: brakedown_num_rows,
            num_brakedown_queries: 0, //compute
            brakedown_row_len: brakedown_row_len,
            brakedown_codeword_len: brakedown_codeword_len,
            basefold: basefold,
            trusted_commits: [],  //to generate using Spark
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
        let num_rows = pp.brakedown_num_rows;
        let codeword_len = pp.brakedown.codeword_len();
        let row_len = pp.brakedown.row_len();
        let (x_0, x_1) = point_to_tensor(num_rows, point);
        let mut combined_codeword = vec![F::ZERO; codeword_len];

        // Taking a linear combination of the rows of the commitment matrix
        for i in 0..num_rows {
            for j in 0..codeword_len {
                combined_codeword[j] += x_0[i] * comm.rows[codeword_len * i + j];
            }
        }   
        
        // Commiting to the message and (codeword - message) parts of comined_codeword
        let mut p: Vec<F> = Vec::new() ; 
        for i in 0..16 {
            p.extend(&combined_codeword[0..row_len]);
        }
        let mut p_prime: Vec<F> = Vec::new() ; 
        let zero_padding: Vec<F> = vec![F::ZERO; codeword_len - num_rows];
        for i in 0..16 {
            p_prime.extend(&combined_codeword[row_len..]);
            p_prime.extend(&zero_padding);
        }
        let p_clone = p.clone();
        let p_prime_clone = p_prime.clone();
        let commitment_to_p = Basefold::<F, H, S>::commit(&pp.basefold, 
                                                        &MultilinearPolynomial::new(p_clone)).unwrap();
        let commitment_to_p_prime = Basefold::<F, H, S>::commit(&pp.basefold, 
                                                        &MultilinearPolynomial::new(p_prime_clone)).unwrap();
        transcript.write_commitment(commitment_to_p.codeword_tree_root());     
        transcript.write_commitment(commitment_to_p_prime.codeword_tree_root());    

        // Proximity test for the commitment matrix
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        let mut col_idx = vec![0 as usize; pp.num_brakedown_queries];
        for i in 0..pp.num_brakedown_queries {
            col_idx[i] = squeeze_challenge_idx(transcript, codeword_len);
            transcript.write_field_elements(comm.rows.iter().skip(col_idx[i]).step_by(codeword_len))?;
            let mut offset = 0;
            for (idx, width) in (1..=depth).rev().map(|depth| 1 << depth).enumerate() {
                let neighbor_idx = (col_idx[i] >> idx) ^ 1;
                transcript.write_commitment(&comm.intermediate_hashes[offset + neighbor_idx])?;
                offset += width;
            }    
        }   

        // Sum-check and Spark are yet to be implemented
        let mut mask = vec![F::ZERO; 2 * row_len];
        for i in 0..pp.num_brakedown_queries {
            mask[col_idx[i]] = transcript.squeeze_challenge();
        }
        let mut p_p_prime = Vec::<F>::new();
        p_p_prime.extend(&p);
        p_p_prime.extend(&p_prime);
        transcript.write_field_elements(&sum_check_prover_round_one(&mask, &p_p_prime));
        for _ in 1..((2 * row_len).next_power_of_two().ilog2() as usize) {
            let r = transcript.squeeze_challenge();
            transcript.write_field_elements(&sum_check_prover_later_round(&mut mask, &mut p_p_prime, r));
        }  

        Ok(())                                      
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
            let (x_0, x_1) = point_to_tensor(num_rows, point);
            let mut combined_codeword = vec![F::ZERO; codeword_len];

            let commitment_to_p = transcript.read_commitment().unwrap();
            let commitment_to_p_prime = transcript.read_commitment().unwrap(); //Just the roots, not BasefoldCommitment objects


            // Read all the queried columns and check their Merkle paths
            let depth = codeword_len.next_power_of_two().ilog2() as usize;
            let mut cols = Vec::<F>::new();
            for _ in 0..vp.num_brakedown_queries {
                let col_idx = squeeze_challenge_idx(transcript, codeword_len);
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
                    if (col_idx >> idx) & 1 == 0 {
                        hasher.update(&output);
                        hasher.update(neighbor);
                    } else {
                        hasher.update(neighbor);
                        hasher.update(&output);
                    }
                    output = hasher.finalize_fixed_reset();
                }
                if &output != &comm.root {
                    return Err(Error::InvalidPcsOpen(
                        "Invalid merkle tree opening".to_string(),
                    ));
                }
            }

            Ok(())
    }
}


fn point_to_tensor<F: PrimeField>(num_rows: usize, point: &[F]) -> (Vec<F>, Vec<F>) {
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(point.len() - num_rows.ilog2() as usize);
    let t_0 = MultilinearPolynomial::eq_xy(lo).into_evals();
    let t_1 = MultilinearPolynomial::eq_xy(hi).into_evals();
    (t_0, t_1)
}

fn squeeze_challenge_idx<F: PrimeField>(
    transcript: &mut impl FieldTranscript<F>,
    cap: usize,
) -> usize {
    let challenge = transcript.squeeze_challenge();
    let mut bytes = [0; size_of::<u32>()];
    bytes.copy_from_slice(&challenge.to_repr().as_ref()[..size_of::<u32>()]);
    u32::from_le_bytes(bytes) as usize % cap
}

fn sum_check_prover_round_one<F: PrimeField>(mask: &Vec<F>, p_p_prime: & Vec<F>) -> Vec<F> {
    let f_2 = F::ONE + F::ONE;
    let f_2_inv = f_2.invert().unwrap();
    let f_3 = f_2 + F::ONE;
    let mask_at_zero: F = mask[0..mask.len()/2].iter().sum();   
    let mask_at_one: F =  mask[mask.len()/2..].iter().sum();    
    let mask_at_two = f_2 * mask_at_one + mask_at_zero;
    let p_p_prime_at_zero: F = p_p_prime[0..mask.len()/2].iter().sum();                 
    let p_p_prime_at_one: F = p_p_prime[mask.len()/2..].iter().sum(); 
    let p_p_prime_at_two = f_2 * p_p_prime_at_one - p_p_prime_at_zero;
    let a_0 = mask_at_zero * p_p_prime_at_zero;
    let a_1 = mask_at_one * p_p_prime_at_one;  
    let a_2 = mask_at_two * p_p_prime_at_two;  
    [a_0 * f_2_inv - a_1 + a_2 * f_2_inv,
                f_2 * a_1 - f_3 * a_0 * f_2_inv - a_2 * f_2_inv, a_0].to_vec()
}

fn sum_check_prover_later_round<F: PrimeField>(mask: &mut Vec<F>, p_p_prime: &mut Vec<F>, r: F) -> Vec<F> {
    let f_2 = F::ONE + F::ONE;
    let f_2_inv = f_2.invert().unwrap();
    let f_3 = f_2 + F::ONE;
    for i in 0..mask.len()/2 {
        mask[i] = (F::ONE - r) * mask[i] + r * mask[i + mask.len()/2];
        p_p_prime[i] = (F::ONE - r) * p_p_prime[i] + r * p_p_prime[i + mask.len()/2];
    }
    mask.resize(mask.len()/2, F::ZERO);
    p_p_prime.resize(mask.len()/2, F::ZERO);

    let mask_at_zero: F = mask[0..mask.len()/2].iter().sum();   
    let mask_at_one: F =  mask[mask.len()/2..].iter().sum();    
    let mask_at_two = f_2 * mask_at_one + mask_at_zero;
    let p_p_prime_at_zero: F = p_p_prime[0..mask.len()/2].iter().sum();                   
    let p_p_prime_at_one: F = p_p_prime[mask.len()/2..].iter().sum();  
    let p_p_prime_at_two = f_2 * p_p_prime_at_one - p_p_prime_at_zero;
    let a_0 = mask_at_zero * p_p_prime_at_zero;
    let a_1 = mask_at_one * p_p_prime_at_one;  
    let a_2 = mask_at_two * p_p_prime_at_two;  
    [a_0 * f_2_inv - a_1 + a_2 * f_2_inv,
                f_2 * a_1 - f_3 * a_0 * f_2_inv - a_2 * f_2_inv, a_0].to_vec()
}



/*
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
        let brakedown_num_rows = COL_SIZE;
        let brakedown = Brakedown::new::<S>(num_vars, 20.min((1 << num_vars) - 1), brakedown_num_rows, rng);

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
            brakedown_num_rows: brakedown_num_rows,
            num_brakedown_queries: 0, //compute

            table_w_weights: table_w_weights,
            table: table,
            num_basefold_queries: S::get_reps(),
            num_basefold_rounds: Some(log2_strict(poly_size) - S::get_basecode_rounds()),
            basefold_log_rate: log_rate,
            rs_basecode: rs_basecode,

            trusted_commits: [],  //to generate using Spark

            rng: test_rng.clone(),
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
        let rng = ChaCha8Rng::from_entropy();           // Use only if we want to have seperate p and q
        let num_rows = pp.brakedown_num_rows;
        let codeword_len = pp.brakedown.codeword_len();
        let (x_0, x_1) = point_to_tensor(num_rows, point);
        let p_and_p_prime = { // if pp.brakedown_num_rows > 1 {
            let combine = |combined_row: &mut [F], coeffs: &[F]| {
                parallelize(combined_row, |(combined_row, offset)| {
                    combined_row
                        .iter_mut()
                        .zip(offset..)
                        .for_each(|(combined, column)| {
                            *combined = F::ZERO;
                            coeffs
                                .iter()
                                .zip(comm.rows.iter().skip(column).step_by(pp.brakedown.codeword_len()))
                                .for_each(|(coeff, eval)| {
                                    *combined += *coeff * eval;
                                });
                        })
                });
            };
            let mut combined_row = vec![F::ZERO; codeword_len];

            // The following can be commented out if we nsure that the point is random and field 128 bits 
            // (for 100 bits of security) because it repeats proximity tests with fresh randomness for 
            // smaller fields.
            /*for _ in 0..pp.brakedown.num_proximity_testing() {
                let coeffs = transcript.squeeze_challenges(num_rows);
                combine(&mut combined_row, &coeffs);
                transcript.write_field_elements(&combined_row)?;
            }*/                                                     
            combine(&mut combined_row, &x_0);
            combined_row
            //Cow::Owned(combined_row)
        }; /*else {
            Cow::Borrowed(comm.rows.as_slice())
        };*/
    }
}

fn point_to_tensor<F: PrimeField>(num_rows: usize, point: &[F]) -> (Vec<F>, Vec<F>) {
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(point.len() - num_rows.ilog2() as usize);
    let t_0 = MultilinearPolynomial::eq_xy(lo).into_evals();
    let t_1 = MultilinearPolynomial::eq_xy(hi).into_evals();
    (t_0, t_1)
}

/*fn combine<F>(combined_row: &mut [F], coeffs: &[F]) {
    parallelize(combined_row, |(combined_row, offset)| {
        combined_row
            .iter_mut()
            .zip(offset..)
            .for_each(|(combined, column)| {
                *combined = F::ZERO;
                coeffs
                    .iter()
                    .zip(comm.rows.iter().skip(column).step_by(pp.brakedown.codeword_len()))
                    .for_each(|(coeff, eval)| {
                        *combined += *coeff * eval;
                    });
            })
    });
}*/*/ 