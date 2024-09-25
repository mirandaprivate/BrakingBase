use crate::pcs::multilinear::{ basefold, brakedown };
use crate::pcs::Commitment;
use crate::piop::sum_check::{self, evaluate};
use crate::piop::sum_check::{
    classic::{ ClassicSumCheck, CoefficientsProver },
    eq_xy_eval,
    SumCheck as _,
    VirtualPolynomial,
};
use crate::util::code::{self, ParityCheckMatrix};
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
use core::fmt::Debug;
use core::{ hash, num };
use core::ptr::addr_of;
use ctr;
use ff::{ derive, BatchInverter };
use generic_array::GenericArray;
use halo2_curves::bn256::{ Bn256, Fr };
use rayon::iter::IntoParallelIterator;
use std::{ collections::HashMap, iter, ops::Deref, time::Instant };

use plonky2_util::{ log2_strict, reverse_bits, reverse_index_bits_in_place };
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
};

use super::brakedown::{MultilinearBrakedownCommitment};


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
    blow_up_factor: usize,
    basefold: BasefoldParams<F>,
    basefold_prover_params: BasefoldProverParams<F>,
    basefold_verifier_params: BasefoldVerifierParams<F>,
    trusted_commits: Vec<BasefoldCommitment<F, H>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrakingbaseProverParams<F: PrimeField> {
    num_vars: usize,
    brakedown: Brakedown<F>, // parity check matrix implicitly provided here
    brakedown_num_rows: usize,
    num_brakedown_queries: usize,
    partity_check_matrix: ParityCheckMatrix<F>,
    blow_up_factor: usize,
    basefold: BasefoldProverParams<F>

}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingbaseVerifierParams<F: PrimeField, H: Hash> {
    num_vars: usize,
    brakedown_num_rows: usize,
    num_brakedown_queries: usize,
    brakedown_row_len: usize,
    brakedown_codeword_len: usize,
    blow_up_factor: usize,
    basefold: BasefoldVerifierParams<F>,
    trusted_commits: Vec<BasefoldCommitment<F, H>>, // replace by Output<H>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct BrakingbaseCommitment<F: PrimeField, H: Hash> {
    rows: Vec<F>,
    intermediate_hashes: Vec<Output<H>>,
    root: Output<H>,
}

impl<F: PrimeField> BrakingbaseProverParams<F> {
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
        let brakedown = Brakedown::new::<S>(
            num_vars,
            (20).min((1 << num_vars) - 1),
            brakedown_num_rows,
            rng
        );
        let brakedown_row_len = brakedown.row_len();
        let brakedown_codeword_len = brakedown.codeword_len();
        let parity_check_matrix = brakedown.parity_check_matrix();
        let blow_up_factor = parity_check_matrix.row.len().next_power_of_two()/brakedown_row_len;

        // Generate BaseFold parameters by running BaseFold's setup algo.
        let mut rng2 = ChaCha8Rng::from_entropy();
        let basefold = Basefold::<F, H, S>
            ::setup((blow_up_factor * poly_size) / COL_SIZE, batch_size, rng2)
            .unwrap();

        
        // Compute the trusted commits
        let (basefold_prover_params, basefold_verifier_params) = Basefold::<F, H, S>
            ::trim(&basefold, poly_size, batch_size)
            .unwrap();
        let mut val = parity_check_matrix.clone().val;
        val.resize((blow_up_factor * poly_size) / COL_SIZE, F::ZERO);
        let mut row = vec![F::ZERO; (blow_up_factor * poly_size) / COL_SIZE];
        let mut col = vec![F::ZERO; (blow_up_factor * poly_size) / COL_SIZE];
        for i in 0..parity_check_matrix.row.len() {
            row[i] = F::try_from(parity_check_matrix.row[i] as u64).unwrap();
        }
        for i in 0..parity_check_matrix.row.len() {
            col[i] = F::try_from(parity_check_matrix.col[i] as u64).unwrap();
        }
        let mut trusted_commits = Vec::<BasefoldCommitment<F, H>>::new();
        trusted_commits.push(Basefold::<F, H, S>
            ::commit(&basefold_prover_params, 
                &MultilinearPolynomial::<F>::new(val))
            .unwrap());
        trusted_commits.push(Basefold::<F, H, S>
            ::commit(&basefold_prover_params, 
                &MultilinearPolynomial::<F>::new(row))
            .unwrap());
        trusted_commits.push(Basefold::<F, H, S>
            ::commit(&basefold_prover_params, 
                &MultilinearPolynomial::<F>::new(col))
            .unwrap());

        Ok(BrakingbaseParams {
            num_vars: num_vars,
            brakedown: brakedown,
            brakedown_num_rows: brakedown_num_rows,
            num_brakedown_queries: 0, //compute
            brakedown_row_len: brakedown_row_len,
            brakedown_codeword_len: brakedown_codeword_len,
            partity_check_matrix: parity_check_matrix,
            blow_up_factor: blow_up_factor,
            basefold: basefold,
            basefold_prover_params: basefold_prover_params,
            basefold_verifier_params: basefold_verifier_params,
            trusted_commits: [].to_vec(), //to generate using Spark
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

        Ok((
            BrakingbaseProverParams {
                num_vars: param.num_vars,
                brakedown: param.brakedown.clone(),
                brakedown_num_rows: param.brakedown_num_rows,
                num_brakedown_queries: 0, //compute
                partity_check_matrix: param.partity_check_matrix.clone(),
                blow_up_factor: param.blow_up_factor,
                basefold: param.basefold_prover_params.clone(),
            },
            BrakingbaseVerifierParams {
                num_vars: param.num_vars,
                brakedown_num_rows: param.brakedown_num_rows,
                num_brakedown_queries: param.num_brakedown_queries,
                brakedown_row_len: param.brakedown_row_len,
                brakedown_codeword_len: param.brakedown_codeword_len,
                blow_up_factor : param.blow_up_factor,
                basefold: param.basefold_verifier_params.clone(),
                trusted_commits: param.trusted_commits.clone(),
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

        // Commiting to the message and (codeword - message) parts of comined_codeword
        let mut p: Vec<F> = Vec::new();

        // The number of coefficients in H is pp.blow_up_factor * row_len.
        for i in 0..pp.blow_up_factor {
            p.extend(&combined_codeword[0..row_len]);
        }
        let mut p_prime: Vec<F> = Vec::new();
        let zero_padding: Vec<F> = vec![F::ZERO; 2 * row_len - codeword_len];
        for i in 0..pp.blow_up_factor {
            p_prime.extend(&combined_codeword[row_len..]);
            p_prime.extend(&zero_padding);
        }
        let p_clone = p.clone();
        let p_prime_clone = p_prime.clone();
        let p_commit = Basefold::<F, H, S>
            ::commit(&pp.basefold, &MultilinearPolynomial::new(p_clone))
            .unwrap();
        let p_prime_commit = Basefold::<F, H, S>
            ::commit(&pp.basefold, &MultilinearPolynomial::new(p_prime_clone))
            .unwrap();
        transcript.write_commitment(p_commit.codeword_tree_root());
        transcript.write_commitment(p_prime_commit.codeword_tree_root());

        // Proximity test for the commitment matrix
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        let mut col_idx = vec![0 as usize; pp.num_brakedown_queries];
        for i in 0..pp.num_brakedown_queries {
            col_idx[i] = squeeze_challenge_idx(transcript, codeword_len);
            transcript.write_field_elements(
                comm.rows.iter().skip(col_idx[i]).step_by(codeword_len)
            )?;
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
        let u = transcript.squeeze_challenges(row_len.ilog2().try_into().unwrap());

        //TODO 2: Realise H(X,u) vector, that is, MLE of the matrix H with Y coordinates replaced by u. This is now a polynomial in X variables.
        let mut h = evaluate_H(&pp.partity_check_matrix, &u, pp.brakedown.codeword_len());
        h.resize(2 * row_len, F::ZERO);

        let mut mask = vec![F::ZERO; 2 * row_len];
        let challenges = transcript.squeeze_challenges(pp.num_brakedown_queries);
        for i in 0..pp.num_brakedown_queries {
            mask[col_idx[i]] = challenges[i];
        }
        let mut p_p_prime = Vec::<F>::new();
        p_p_prime.extend(&p[0..row_len]);
        p_p_prime.extend(&p_prime[0..row_len]);

        //TODO 4: Sample two random points here.
        let random_combiners = transcript.squeeze_challenges(2);

        let sum_check_rounds = (2 * row_len).next_power_of_two().ilog2() as usize;
        let mut first_sum_check_random_points = vec![F::ZERO; sum_check_rounds];

        let h_clone = h.clone();
        let p_clone = p.clone();
        let p_prime_clone = p_prime.clone();

        //TODO 5: Make a function called first_sum_check_prover(). Call the function here with p_p_prime, mask, H(X,U), the two random points, and transcript as input here.

        /* first_sum_check_prover() description: does the entire code of the sum_check (all the rounds) and 
        the messages are included in the transcript within the function. This function takes 
        transcript as mutable reference. 
        For reference see the sum-check implemented here https://github.com/arithmic/Dual_PCS/blob/main/Spartan/Spartan_with_gkr/src/prover/batch_eval.rs
        The len_4 interpolate here will be replaced by the expression you use at the end of  sum_check_prover_round_one or sum_check_prover_later_round
        */

        //first_sum_check_prover(sum_check_rounds, p_p_prime, mask, h, random_combiners, &mut first_sum_check_random_points, transcript);
        first_sum_check_prover::<F, H, S>(
            sum_check_rounds,
            p_p_prime,
            mask,
            h,
            random_combiners,
            &mut first_sum_check_random_points,
            transcript
        );

        //TODO 3: evaluate h, p, p_prime at first_sum_check_random_points
        let h_eval = evaluate_poly(&h_clone, &first_sum_check_random_points);
        let p_eval = partial_evaluate_poly(&p_clone, &first_sum_check_random_points, 1);
        let p_prime_eval = partial_evaluate_poly(&p_prime_clone, &first_sum_check_random_points, 1);
        transcript.write_field_elements([h_eval, p_eval, p_prime_eval].iter());

        //TODO 6.1: Commit to H_erow, H_ecol using Basefold
        //TODO 6.2(Bhargav): Compute H_val -- Check sum_check_rounds
        let mut h_val = pp.partity_check_matrix.val.clone();
        h_val.resize(h_val.len().next_power_of_two(), F::ZERO);
        let mut h_erow = compute_oracle_poly(&pp.partity_check_matrix.row, 
            &first_sum_check_random_points);
        h_erow.resize(h_erow.len().next_power_of_two(), F::ZERO);
        let mut h_ecol = compute_oracle_poly(&pp.partity_check_matrix.row, 
            &u);
        h_ecol.resize(h_ecol.len().next_power_of_two(), F::ZERO);

        let h_erow_commit = Basefold::<F, H, S>
            ::commit(&pp.basefold, &MultilinearPolynomial::new(h_erow.clone()))
            .unwrap();
        let h_ecol_commit = Basefold::<F, H, S>
            ::commit(&pp.basefold, &MultilinearPolynomial::new(h_ecol.clone()))
            .unwrap();
        transcript.write_commitment(h_erow_commit.codeword_tree_root());
        transcript.write_commitment(h_ecol_commit.codeword_tree_root());
        
        let sum_check_rounds = (h_val.len()).next_power_of_two().ilog2() as usize;
        let mut second_sum_check_random_points = vec![F::ZERO; sum_check_rounds];

        let h_val_clone = h_val.clone();
        let h_erow_clone = h_erow.clone();
        let h_ecol_clone = h_ecol.clone();    

        //TODO 7: Make a function called second_sum_check_prover(). Call the function here with H_erow, H_ecol, H_val, and transacript
        /* second_sum_check_prover() description: does the entire code of the sum_check (all the rounds) and 
        the messages are included in the transcript within the function. This function takes 
        transcript as mutable reference. 
        For reference see the sum-check implemented here https://github.com/arithmic/Dual_PCS/blob/main/Spartan/Spartan_with_gkr/src/prover/batch_eval.rs.
        The sum-check expression is H_erow\cdot H_ecol \cdot H_eval, and hence would need len_4 interpolate.
        */
        second_sum_check_prover::<F, H, S>(
            sum_check_rounds,
            h_erow,
            h_ecol,
            h_val,
            &mut second_sum_check_random_points,
            transcript
        );

        transcript.write_field_element(&evaluate_poly(&h_val_clone, 
            &second_sum_check_random_points));
        transcript.write_field_element(&evaluate_poly(&h_erow_clone, 
            &second_sum_check_random_points));
        transcript.write_field_element(&evaluate_poly(&h_ecol_clone, 
            &second_sum_check_random_points));

        //TODO 8: Incorporate GKR from https://github.com/arithmic/Dual_PCS/tree/main/Grand_product/grand_product_with_gkr to our code.
        //This might need some work, and we might have to sit down with Ashish for this.
        // We could alternatively also implement Quarks.
        //Call the grand-product check argument. In total we would have 4 grand-product checks.

        //TODO 9: Batch Evaluate

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
        let mut combined_codeword = vec![F::ZERO; codeword_len];

        let commitment_to_p = transcript.read_commitment().unwrap();
        let commitment_to_p_prime = transcript.read_commitment().unwrap(); //Just the roots, not BasefoldCommitment objects

        // Read all the queried columns and check their Merkle paths
        let depth = codeword_len.next_power_of_two().ilog2() as usize;
        //let sum_check_val = F::ZERO;
        let mut col_idx = vec![0 as usize; vp.num_brakedown_queries];
        let mut cols = vec![F::ZERO; vp.num_brakedown_queries * vp.brakedown_row_len];
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

        let mut sum_check_val = F::ZERO;
        let challenges = transcript.squeeze_challenges(vp.num_brakedown_queries);
        for j in 0..vp.num_brakedown_queries {
            for i in 0..vp.brakedown_row_len {
                sum_check_val += challenges[j] * x_0[i] * cols[j * vp.brakedown_row_len + j];  // make x_1[i]
            }
        }

        let mut a = transcript.read_field_elements(3).unwrap();
        if sum_check_val != (F::ONE + F::ONE) * a[0] + a[1] + a[2] {
            println!("HERE");
            return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
            println!("NOT HERE");
        }
        for _ in 1..(2 * row_len).next_power_of_two().ilog2() as usize {
            let r = transcript.squeeze_challenge();
            sum_check_val = a[0] + a[1] * r + a[2] * r * r;
            a = transcript.read_field_elements(3).unwrap();
            if sum_check_val != (F::ONE + F::ONE) * a[0] + a[1] + a[2] {
                return Err(Error::InvalidPcsOpen("Sum check failed".to_string()));
            }
        }

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

fn point_to_tensor<F: PrimeField>(num_rows: usize, point: &[F]) -> (Vec<F>, Vec<F>) {
    assert!(num_rows.is_power_of_two());
    let (hi, lo) = point.split_at(point.len() - (num_rows.ilog2() as usize));
    let t_0 = MultilinearPolynomial::eq_xy(lo).into_evals();
    let t_1 = MultilinearPolynomial::eq_xy(hi).into_evals();
    (t_0, t_1)
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

fn sum_check_prover_round_one<F: PrimeField>(mask: &Vec<F>, p_p_prime: &Vec<F>) -> Vec<F> {
    let f_2 = F::ONE + F::ONE;
    let f_2_inv = f_2.invert().unwrap();
    let f_3 = f_2 + F::ONE;
    let mask_at_zero: F = mask[0..mask.len() / 2].iter().sum();
    let mask_at_one: F = mask[mask.len() / 2..].iter().sum();
    let mask_at_two = f_2 * mask_at_one + mask_at_zero;
    let p_p_prime_at_zero: F = p_p_prime[0..mask.len() / 2].iter().sum();
    let p_p_prime_at_one: F = p_p_prime[mask.len() / 2..].iter().sum();
    let p_p_prime_at_two = f_2 * p_p_prime_at_one - p_p_prime_at_zero;
    let a_0 = mask_at_zero * p_p_prime_at_zero;
    let a_1 = mask_at_one * p_p_prime_at_one;
    let a_2 = mask_at_two * p_p_prime_at_two;
    [
        a_0 * f_2_inv - a_1 + a_2 * f_2_inv,
        f_2 * a_1 - f_3 * a_0 * f_2_inv - a_2 * f_2_inv,
        a_0,
    ].to_vec()
}

fn sum_check_prover_later_round<F: PrimeField>(
    mask: &mut Vec<F>,
    p_p_prime: &mut Vec<F>,
    r: F
) -> Vec<F> {
    let f_2 = F::ONE + F::ONE;
    let f_2_inv = f_2.invert().unwrap();
    let f_3 = f_2 + F::ONE;
    for i in 0..mask.len() / 2 {
        mask[i] = (F::ONE - r) * mask[i] + r * mask[i + mask.len() / 2];
        p_p_prime[i] = (F::ONE - r) * p_p_prime[i] + r * p_p_prime[i + mask.len() / 2];
    }
    mask.resize(mask.len() / 2, F::ZERO);
    p_p_prime.resize(p_p_prime.len() / 2, F::ZERO);

    let mask_at_zero: F = mask[0..mask.len() / 2].iter().sum();
    let mask_at_one: F = mask[mask.len() / 2..].iter().sum();
    let mask_at_two = f_2 * mask_at_one + mask_at_zero;
    let p_p_prime_at_zero: F = p_p_prime[0..mask.len() / 2].iter().sum();
    let p_p_prime_at_one: F = p_p_prime[mask.len() / 2..].iter().sum();
    let p_p_prime_at_two = f_2 * p_p_prime_at_one - p_p_prime_at_zero;
    let a_0 = mask_at_zero * p_p_prime_at_zero;
    let a_1 = mask_at_one * p_p_prime_at_one;
    let a_2 = mask_at_two * p_p_prime_at_two;
    [
        a_0 * f_2_inv - a_1 + a_2 * f_2_inv,
        f_2 * a_1 - f_3 * a_0 * f_2_inv - a_2 * f_2_inv,
        a_0,
    ].to_vec()
}

//first_sum_check_prover(). Call the function here with p_p_prime, mask, H(X,U), the two random points, and transcript as input here.
pub fn first_sum_check_prover<F,H, S>(
    sum_check_rounds: usize,
    mut p_p_prime: Vec<F>,
    mut mask:  Vec<F>,
    mut h: Vec<F>,
    random_combiners: Vec<F>,
    first_sum_check_random_points: &mut Vec<F>,
    transcript: &mut impl TranscriptWrite<<Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>
)
where F: PrimeField + Serialize + DeserializeOwned, H: Hash, S: BrakingbaseSpec
 {
    //transcript.write_field_elements(&sum_check_prover_round_one(&mask, &p_p_prime));
    for i in 0..sum_check_rounds {
        let f_2 = F::ONE + F::ONE;
        let f_2_inv = f_2.invert().unwrap();
        let f_3 = f_2 + F::ONE;

        let mask_at_zero: F = mask[0..mask.len() / 2].iter().sum();
        let mask_at_one: F = mask[mask.len() / 2..].iter().sum();
        let mask_at_two = f_2 * mask_at_one - mask_at_zero;
        
        let p_p_prime_at_zero: F = p_p_prime[0..mask.len() / 2].iter().sum();
        let p_p_prime_at_one: F = p_p_prime[mask.len() / 2..].iter().sum();
        let p_p_prime_at_two = f_2 * p_p_prime_at_one - p_p_prime_at_zero;

        let h_at_zero: F = h[0..mask.len() / 2].iter().sum();
        let h_at_one: F = h[mask.len() / 2..].iter().sum();
        let h_at_two = f_2 * h_at_one - h_at_zero;

        let a_0 = (random_combiners[0]* mask_at_zero + random_combiners[1]* h_at_zero) * p_p_prime_at_zero;
        let a_1 = (random_combiners[0]* mask_at_one + random_combiners[1]* h_at_one) * p_p_prime_at_one;
        let a_2 = (random_combiners[0]* mask_at_two + random_combiners[1]* h_at_two) * p_p_prime_at_two;
        let polynomial_current_round = [
            a_0 * f_2_inv - a_1 + a_2 * f_2_inv,
            -f_3 * a_0 * f_2_inv + f_2 * a_1 - a_2 * f_2_inv,
            a_0,
        ].to_vec();
        transcript.write_field_elements(
            &polynomial_current_round
        );
        let r = transcript.squeeze_challenge();
        first_sum_check_random_points[i] = r;

        for i in 0..mask.len() / 2 {
            mask[i] = (F::ONE - r) * mask[i] + r * mask[i + mask.len() / 2];
            p_p_prime[i] = (F::ONE - r) * p_p_prime[i] + r * p_p_prime[i + p_p_prime.len() / 2];
            h[i] = (F::ONE - r) * h[i] + r * h[i + h.len() / 2];
        }
        mask.resize(mask.len() / 2, F::ZERO);
        p_p_prime.resize(p_p_prime.len() / 2, F::ZERO);
        h.resize(h.len() / 2, F::ZERO);
        
    }
}

//second_sum_check_prover(). Call the function here with H_val. H_erow, H_ecol, and transcript as input here.
pub fn second_sum_check_prover<F,H, S>(
    sum_check_rounds: usize,
    mut h_erow: Vec<F>,
    mut h_ecol:  Vec<F>,
    mut h_val: Vec<F>,
    second_sum_check_random_points: &mut Vec<F>,
    transcript: &mut impl TranscriptWrite<<Brakingbase<F, H, S> as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>
)
where F: PrimeField + Serialize + DeserializeOwned, H: Hash, S: BrakingbaseSpec
 {
    //transcript.write_field_elements(&sum_check_prover_round_one(&mask, &p_p_prime));
    for i in 0..sum_check_rounds {
        let f_2 = F::ONE + F::ONE;
        let f_2_inv = f_2.invert().unwrap();
        let f_3 = f_2 + F::ONE;

        let h_erow_at_zero: F = h_erow[0..h_erow.len() / 2].iter().sum();
        let h_erow_at_one: F = h_erow[h_erow.len() / 2..].iter().sum();
        let h_erow_at_two = f_2 * h_erow_at_one - h_erow_at_zero;
        let h_erow_at_minus_one = f_2 * h_erow_at_zero - h_erow_at_one;
        
        let h_ecol_at_zero: F = h_ecol[0..h_ecol.len() / 2].iter().sum();
        let h_ecol_at_one: F = h_ecol[h_ecol.len() / 2..].iter().sum();
        let h_ecol_at_two = f_2 * h_ecol_at_one - h_ecol_at_zero;
        let h_ecol_at_minus_one = f_2 * h_ecol_at_zero - h_ecol_at_one;

        let h_val_at_zero: F = h_val[0..h_val.len() / 2].iter().sum();
        let h_val_at_one: F = h_val[h_val.len() / 2..].iter().sum();
        let h_val_at_two = f_2 * h_val_at_one - h_val_at_zero;
        let h_val_at_minus_one = f_2 * h_val_at_zero - h_val_at_one;

        let a_0 = h_erow_at_zero * h_ecol_at_zero * h_val_at_zero;
        let a_1 = h_erow_at_one * h_ecol_at_one * h_val_at_one;
        let a_2 = h_erow_at_two * h_ecol_at_two * h_val_at_two;
        let a_minus_one = h_erow_at_minus_one * h_ecol_at_minus_one * h_val_at_minus_one;

        //TODO (Bhargav): edit the following expression to derive the 4 coefficients
        let f_2 = F::ONE + F::ONE;
        let f_2_inv = f_2.invert().unwrap();
        let f_3 = f_2 + F::ONE;
        let f_3_inv = f_3.invert().unwrap();
        let f_6 = f_3 + f_3;
        let f_6_inv = f_6.invert().unwrap();
        let polynomial_current_round = [
            a_0 * f_2_inv  - a_1 * f_2_inv + a_2 * f_6_inv - a_minus_one * f_6_inv,
            -a_0 + a_1 * f_2_inv + a_minus_one * f_2_inv,
            -a_0 * f_2_inv  + a_1 - a_2 * f_6_inv - a_minus_one * f_3_inv,
            a_0
        ].to_vec();
        transcript.write_field_elements(
            &polynomial_current_round
        );
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

fn evaluate_H<F: PrimeField> (H: &ParityCheckMatrix<F>, u: &Vec<F>, size: usize) -> Vec<F> {
    let mut H_at_u = vec![F::ZERO; size];
    let tensor_u = point_to_tensor(1, u).1;
    for i in 0..H.row.len() {
        H_at_u[H.row[i]] += H.val[i] * tensor_u[H.col[i]];
    }
    H_at_u
}

fn evaluate_poly<F: PrimeField> (coeffs: &Vec<F>, point: &Vec<F>) -> F {
    let mut eval = F::ZERO;
    let tensor_point = point_to_tensor(1, point).1;
    for i in 0..tensor_point.len() {
        eval += coeffs[i] * tensor_point[i];
    }
    eval
}

fn partial_evaluate_poly<F: PrimeField> (coeffs: &Vec<F>, point: &Vec<F>, skip: usize) -> F {
    let mut eval = F::ZERO;
    let tensor_point = point_to_tensor(point.len() - (1 << skip), point).0;
    for i in 0..tensor_point.len() {
        eval += coeffs[i] * tensor_point[i];
    }
    eval
}

fn compute_oracle_poly<F: PrimeField> (coeffs: &Vec<usize>, point: &Vec<F>) -> Vec<F> {
    let mut oracle_poly = vec![F::ZERO; coeffs.len()];
    for i in 0..coeffs.len() {
        assert!(coeffs[i] < 1 << (point.len() + 1));
        let mut tmp = coeffs[i];
        for j in 1..=point.len() {
            let bit = tmp - ((tmp >> 1) << 1); 
            oracle_poly[i] += F::try_from(bit as u64).unwrap() * point[point.len() - j];
            tmp = tmp >> 1;
        }
    }
    oracle_poly
}

#[cfg(test)]
mod test {
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

    use super::{ Brakingbase, BrakingbaseSpec };

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

    fn test_parity_check_matrix () {
        let num_vars = 16;

        let batch_size = 1;
        let mut rng = ChaCha8Rng::from_entropy();

        let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
        let mut parity_check_matrix = vec![vec![Fr::ZERO; params.brakedown_codeword_len - params.brakedown_row_len]; 
            params.brakedown_codeword_len];

        for i in 0..params.partity_check_matrix.row.len() {
            let row = params.partity_check_matrix.row[i];
            let col = params.partity_check_matrix.col[i];
            parity_check_matrix[row][col] = params.partity_check_matrix.val[i];
        }

        let mut rng = ChaCha8Rng::from_entropy();
        let mut msg = vec![Fr::random(&mut rng); params.brakedown_row_len];
        msg.extend(vec![Fr::ZERO; params.brakedown_codeword_len - params.brakedown_row_len]);
        params.brakedown.encode(&mut msg);
        let mut res = vec_matrix_prod(&msg, &parity_check_matrix);

        // for i in 0..res.len() {
        //     res[i] = res[i] - msg[params.brakedown_row_len + i];
        // } 
        //println!("{:?}", msg);
        println!("{:?}", res);
        //println!("{:?}", parity_check_matrix);
    }

    #[test]
    fn test_setup () {
        let num_vars = 15;
        let batch_size = 1;
        let mut rng = ChaCha8Rng::from_entropy();

        let params = Pcs::setup(1 << num_vars, batch_size, rng).unwrap();
        
        println!("{}, {}, {}, {}", params.num_vars, params.brakedown_row_len, 
                params.brakedown_num_rows, params.brakedown_codeword_len);
        println!("{}, {}", params.blow_up_factor, params.basefold.num_vars);
        //println!("{:?}", params.brakedown);
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
        //println!("{}", comm.rows.len());
        //println!("{:?}", comm.rows);
        let mut i = comm.rows.len() - 1;
        while comm.rows[i] == Fr::ONE {
            i -= 1;
        }
    }

    #[test]
    fn test_open() {
        run_commit_open_verify::<_, Pcs, Blake2sTranscript<_>>();
    }


    fn vec_matrix_prod<F: PrimeField>  (vc: &Vec::<F>, mat: &Vec<Vec::<F>>) -> Vec::<F>{
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
