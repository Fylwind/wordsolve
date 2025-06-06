#![crate_type = "cdylib"]
#![crate_type = "lib"]

use float_ord::FloatOrd;
use fxhash::FxHashMap as HashMap;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use static_assertions::const_assert;
use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::io::Write;
use std::rc::Rc;
use std::{cmp, fmt, io, str, time};

#[cfg(target_arch = "wasm32")]
mod js;

fn error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Other, message)
}

lazy_static! {
    static ref REFERENCE_INSTANT: time::Instant = time::Instant::now();
}

#[cfg(not(target_arch = "wasm32"))]
fn now() -> f64 {
    let reference = *REFERENCE_INSTANT; // This must run first!
    time::Instant::now().duration_since(reference).as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
fn now() -> f64 {
    js::now() / 1e3
}

const NUM_CHARS: usize = 5;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct WordChar(u8);

impl WordChar {
    const INVALID: Self = WordChar(0);
}

impl TryFrom<char> for WordChar {
    type Error = io::Error;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        let u = u8::try_from(u32::from(c))
            .map_err(|_| error(&format!("invalid character: {:?}", c)))?;
        if u == 0 {
            return Err(error(&format!("invalid character: {:?}", c)));
        }
        Ok(Self(u))
    }
}

fn word_chars(word: &str) -> io::Result<[WordChar; NUM_CHARS]> {
    let mut result = [WordChar::INVALID; NUM_CHARS];
    for (c, r) in word.chars().zip(&mut result) {
        *r = WordChar::try_from(c)?;
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Color(u8);

impl Color {
    const GRAY: Self = Color(0);
    const YELLOW: Self = Color(1);
    const GREEN: Self = Color(2);
    const COUNT: usize = 3;
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Response(u8);

impl fmt::Debug for Response {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Response::try_from({:?}).unwrap()", format!("{}", self))
    }
}

impl fmt::Display for Response {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for c in Vec::from(*self) {
            write!(f, "{}", c.0)?;
        }
        Ok(())
    }
}

impl TryFrom<&str> for Response {
    type Error = io::Error;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        if s.len() != NUM_CHARS {
            return Err(error(&format!(
                "response must have {} characters",
                NUM_CHARS
            )));
        }
        const_assert!(Color::COUNT as u32 as usize == Color::COUNT);
        s.chars()
            .map(|c| {
                let digit = c
                    .to_digit(Color::COUNT as u32)
                    .ok_or(error("response must consist of numerical digits"))?;
                Ok(Color(digit as _))
            })
            .collect()
    }
}

impl From<Response> for Vec<Color> {
    fn from(code: Response) -> Self {
        let Response(mut code) = code;
        let count = Color::COUNT as u8;
        let mut colors = vec![Color::default(); NUM_CHARS];
        for c in &mut colors {
            c.0 = code % count;
            code /= count;
        }
        colors
    }
}

impl FromIterator<Color> for Response {
    fn from_iter<T: IntoIterator<Item = Color>>(iter: T) -> Self {
        let mut u = 0;
        const_assert!(usize::pow(Color::COUNT, NUM_CHARS as u32) <= u8::MAX as usize);
        for (i, Color(c)) in iter.into_iter().enumerate() {
            u += (c as u8) * u8::pow(Color::COUNT as u8, i as u32);
        }
        Self(u)
    }
}

impl Response {
    fn compute(query: &str, candidate: &str) -> Self {
        let query = word_chars(query).unwrap();
        let candidate = word_chars(candidate).unwrap();
        Self::compute_with(query, candidate)
    }

    fn compute_with(query: [WordChar; NUM_CHARS], mut candidate: [WordChar; NUM_CHARS]) -> Self {
        let mut color_buf = [Color::GRAY; NUM_CHARS];
        for ((color, &query_char), candidate_char) in
            color_buf.iter_mut().zip(&query).zip(&mut candidate)
        {
            if query_char == *candidate_char {
                *color = Color::GREEN;
                *candidate_char = WordChar::INVALID;
            }
        }
        for (color, &query_char) in color_buf.iter_mut().zip(&query) {
            if *color == Color::GRAY {
                for candidate_char in &mut candidate {
                    if query_char == *candidate_char {
                        *color = Color::YELLOW;
                        *candidate_char = WordChar::INVALID;
                        break;
                    }
                }
            }
        }
        color_buf.iter().copied().collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WordId(pub u16);

impl From<usize> for WordId {
    fn from(i: usize) -> Self {
        Self(i.try_into().expect("WordId overflow"))
    }
}

#[derive(Clone, Debug)]
pub struct Matrix {
    responses: Vec<Response>,
    num_candidates: usize,
    pub words: Vec<String>,
    pub word_ids: HashMap<String, WordId>,
}

impl Matrix {
    pub fn build(database: &Database, progress_sink: &mut ProgressSink) -> io::Result<Self> {
        let mut responses = Vec::default();
        let words: Vec<String> = database
            .candidates
            .iter()
            .chain(&database.noncandidates)
            .cloned()
            .collect();
        let word_ids: HashMap<String, WordId> = words
            .iter()
            .enumerate()
            .map(|(i, word)| {
                (
                    word.clone(),
                    WordId(i.try_into().expect("WordId too large")),
                )
            })
            .collect();
        let queries: Vec<[WordChar; NUM_CHARS]> = words
            .iter()
            .map(|word| word_chars(word))
            .collect::<io::Result<Vec<_>>>()?;
        let candidates: Vec<[WordChar; NUM_CHARS]> = database
            .candidates
            .iter()
            .map(|word| word_chars(word))
            .collect::<io::Result<Vec<_>>>()?;
        let mut progress_tracker = ProgressTracker::new(progress_sink);
        for (i, &query) in queries.iter().enumerate() {
            progress_tracker
                .tick(&mut || (i as f64 / queries.len() as f64, format!("{}", words[i])));
            for &candidate in &candidates {
                let response = Response::compute_with(query, candidate);
                responses.push(response)
            }
        }
        Ok(Self {
            responses,
            num_candidates: database.candidates.len(),
            words,
            word_ids,
        })
    }

    fn response(&self, query: WordId, candidate: WordId) -> Response {
        assert!((candidate.0 as usize) < self.num_candidates);
        self.responses[query.0 as usize * self.num_candidates + candidate.0 as usize]
    }

    pub fn word(&self, WordId(i): WordId) -> &str {
        &self.words[i as usize]
    }

    fn words(&self, words: &[WordId]) -> Vec<&str> {
        words.iter().map(|&w| self.word(w)).collect()
    }

    pub fn word_ids(&self, words: &[String]) -> Vec<WordId> {
        words
            .iter()
            .map(|word| *self.word_ids.get(word).expect("word not found"))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Database {
    /// Words that are valid as a solution.
    pub candidates: Vec<String>,
    /// Words that are valid as input, but never as a solution.
    pub noncandidates: Vec<String>,
}

impl Database {
    pub fn parse(s: &str) -> io::Result<Self> {
        let mut candidates = Vec::default();
        let mut noncandidates = Vec::default();
        for line in s.lines() {
            let word = line.trim().to_lowercase();
            let is_noncandidate = word.starts_with("!");
            let word = word.strip_prefix('!').unwrap_or(&word);
            let chars: Vec<char> = word.chars().collect();
            if !(chars.len() == NUM_CHARS && chars.iter().all(char::is_ascii_lowercase)) {
                continue;
            }
            if is_noncandidate {
                &mut noncandidates
            } else {
                &mut candidates
            }
            .push(word.to_owned());
        }
        Ok(Database {
            candidates,
            noncandidates,
        })
    }
}

pub fn parse_guesses<'a>(s: &'a str, separator: &str) -> io::Result<Vec<(&'a str, Response)>> {
    let mut guesses = Vec::default();
    for entry in s.split_terminator(separator) {
        if entry.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = entry.trim().split_whitespace().collect();
        if fields.len() != 2 {
            return Err(error("each line of guesses must have exactly 2 fields"));
        }
        guesses.push((fields[0], Response::try_from(fields[1])?));
    }
    Ok(guesses)
}

pub fn filter_candidates(guesses: &[(&str, Response)], candidates: &[String]) -> Vec<String> {
    candidates
        .iter()
        .cloned()
        .filter(|candidate| {
            guesses
                .iter()
                .all(|&(query, response)| Response::compute(query, candidate) == response)
        })
        .collect()
}

fn ceil_log(mut n: i32, q: i32) -> i32 {
    let mut l = 0;
    while n > 1 {
        n = n / q + (n % q != 0) as i32;
        l += 1;
    }
    l
}

fn render_duration_secs(secs: f64) -> String {
    if !secs.is_finite() {
        format!("??")
    } else if secs >= 86400.0 {
        format!("{:.0}d", secs / 86400.0)
    } else if secs >= 3600.0 {
        format!("{:.0}h", secs / 3600.0)
    } else if secs >= 60.0 {
        format!("{:.0}min", secs / 60.0)
    } else {
        format!("{:.0}s", secs)
    }
}

type Outcomes = Vec<(Response, SmallVec<[WordId; 8]>)>;

fn subdivide_candidates(
    matrix: &Matrix,
    query: WordId,
    candidates: &[WordId],
) -> (f64, Outcomes, bool) {
    let mut outcomes = HashMap::default();
    let mut has_exact = false;
    for &candidate in candidates {
        has_exact |= candidate == query;
        let response = matrix.response(query, candidate);
        let candidates = outcomes.entry(response).or_insert(SmallVec::default());
        candidates.push(candidate);
    }
    let mut outcomes: Outcomes = outcomes.into_iter().collect();
    outcomes
        .sort_unstable_by_key(|(response, candidates)| (cmp::Reverse(candidates.len()), *response));

    // entropy_remaining quantifies is the expected information needed to
    // identify the correct word after partitioning subcandidates by outcomes.
    //
    // We assume the following joint probability distribution:
    //
    //     Pr(word, outcome) = {
    //       1 / n,   if word is in the subcandidates of outcome;
    //       0,       otherwise;
    //     }
    //
    // entropy_remaining can be derived from the conditional entropy:
    //
    //     H(Word | Outcome)
    //       = -SUM[word, outcome] Pr(word, outcome) ln(Pr(word | outcome))
    //       = -SUM[outcome] (k[outcome] / n) ln(1 / k[outcome])
    //       = SUM[outcome] k[outcome] ln(k[outcome]) / n
    //
    // This can also be interpreted as the expectation over all outcomes of
    // the entropy of each subcandidate set (i.e. ln(k[outcome])).
    let n = candidates.len(); // Number of words.
    let entropy_remaining = outcomes
        .iter()
        .map(|(_, subcandidates)| {
            let k = subcandidates.len() as f64; // Number of subcandidates.
            k * f64::ln(k)
        })
        .sum::<f64>()
        / n as f64;

    (f64::exp(entropy_remaining), outcomes, has_exact)
}

fn rank_queries(
    matrix: &Matrix,
    queries: &[WordId],
    candidates: &[WordId],
) -> (Vec<WordId>, Vec<(WordId, f64, Outcomes)>) {
    let mut viable_queries = Vec::default();
    assert!(candidates.len() > 1);
    let mut query_outcomes = Vec::default();
    let mut num_has_exacts: u32 = 0;
    let prefix = if candidates.len() <= 4 {
        candidates
    } else {
        &[]
    };
    for (i, &query) in prefix.into_iter().chain(queries).enumerate() {
        if i >= prefix.len() && prefix.contains(&query) {
            continue;
        }
        let response = matrix.response(query, candidates[0]);
        let mut viable = false;
        for &candidate in &candidates[1..] {
            if matrix.response(query, candidate) != response {
                viable_queries.push(query);
                viable = true;
                break;
            }
        }
        if !viable {
            continue;
        }
        let (k_avg, outcomes, has_exact) = subdivide_candidates(matrix, query, candidates);
        let num_outcomes = outcomes.len();
        num_has_exacts += has_exact as u32;
        if num_outcomes > 1 || candidates[0] == query {
            query_outcomes.push((query, k_avg, outcomes));
            if num_outcomes == candidates.len()
                && (num_has_exacts >= candidates.len() as u32 || has_exact)
            {
                break; // This query is optimal
            }
        }
    }
    // TODO: Prune queries that don't progress fast enough to reach depth_max
    query_outcomes.sort_unstable_by_key(|(query, k_avg, outcomes)| {
        (outcomes.first().unwrap().1.len(), FloatOrd(*k_avg), *query)
    });
    (viable_queries, query_outcomes)
}

type ProgressSink<'a> = dyn FnMut(f64, String) + 'a;

fn tree_progress(stack: &[(f64, f64)]) -> f64 {
    stack.iter().rfold(0.0, |subprogress, &(progress, weight)| {
        progress + weight * subprogress
    })
}

struct ProgressTracker<'a> {
    progress_sink: &'a mut ProgressSink<'a>,
    ticks_before_render: u32,
    ticks_total: u64,
    render_interval_ticks: u32,
    t_start: f64,
}

impl<'a> ProgressTracker<'a> {
    const RENDER_INTERVAL: time::Duration = time::Duration::from_millis(50);

    fn new(progress_sink: &'a mut ProgressSink<'a>) -> Self {
        Self {
            progress_sink,
            ticks_before_render: 1,
            ticks_total: 1,
            render_interval_ticks: 1,
            t_start: now(),
        }
    }

    fn tick(&mut self, query_progress: &mut dyn FnMut() -> (f64, String)) {
        self.ticks_before_render -= 1;
        if self.ticks_before_render != 0 {
            return;
        }
        let (progress, message) = query_progress();
        let elapsed = now() - self.t_start;
        let remaining = elapsed * (1.0 - progress) / progress;
        self.render_interval_ticks = ((Self::RENDER_INTERVAL.as_secs_f64() / elapsed
            * self.ticks_total as f64) as u32)
            .clamp(
                self.render_interval_ticks / 2,
                self.render_interval_ticks * 2,
            )
            .clamp(1, 1000000);
        self.ticks_before_render = self.render_interval_ticks;
        self.ticks_total += self.render_interval_ticks as u64;
        (self.progress_sink)(
            progress,
            format!(
                "{} elapsed | {} left | tbr:{} / {}",
                render_duration_secs(elapsed),
                render_duration_secs(remaining),
                self.ticks_before_render,
                message,
            ),
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Score {
    pub depth_max: u8,
    pub depth_sum: u32,
    pub num_candidates: u16,
}

impl Score {
    const WIN: Self = Self {
        depth_max: 0,
        depth_sum: 0,
        num_candidates: 1,
    };

    const LOSS: Self = Self {
        depth_max: u8::MAX,
        depth_sum: u32::MAX,
        num_candidates: 0,
    };

    fn is_loss(self) -> bool {
        self.depth_max == Self::LOSS.depth_max
    }

    pub fn render(self) -> Vec<String> {
        vec![
            format!("{}", self.depth_max),
            format!(
                "{:.3} ({}/{})",
                self.depth_sum as f64 / self.num_candidates as f64,
                self.depth_sum,
                self.num_candidates,
            ),
        ]
    }

    fn ord(self) -> (cmp::Reverse<u8>, cmp::Reverse<f64>) {
        (
            cmp::Reverse(self.depth_max),
            cmp::Reverse(self.depth_sum as f64 / self.num_candidates as f64),
        )
    }

    fn add_outcome(&mut self, score: Self) {
        self.depth_max = cmp::max(self.depth_max, score.depth_max);
        self.depth_sum += score.depth_sum;
        self.num_candidates += score.num_candidates;
    }

    fn ascend(mut self) -> Self {
        self.depth_max = self.depth_max.saturating_add(1);
        self.depth_sum = self
            .depth_sum
            .checked_add(self.num_candidates as u32)
            .unwrap();
        self
    }
}

#[derive(Clone, Debug)]
pub struct Strategy {
    pub query: WordId,
    pub outcomes: Vec<(Response, Option<Rc<Strategy>>)>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Solver {
    pub dk_trunc: f64,
}

impl Solver {
    pub fn solve(
        &self,
        matrix: &Matrix,
        queries: &[WordId],
        candidates: &[WordId],
        progress_sink: &mut ProgressSink,
    ) -> (Score, Option<Rc<Strategy>>) {
        let mut context = SolverContext {
            state: SolverState::default(),
            conf: self,
            matrix,
            progress_tracker: ProgressTracker::new(progress_sink),
        };
        let solution = context.solve(queries, candidates, u8::MAX);
        progress_sink(1.0, format!(""));
        solution
    }
}

#[derive(Clone, Debug)]
enum CachedResult {
    DepthExceeds(u8),
    Solved {
        score: Score,
        strategy: Option<Rc<Strategy>>,
    },
}

#[derive(Clone, Debug, Default)]
struct SolverState {
    path: Vec<WordId>,
    // Transposition table.
    table: HashMap<Vec<WordId>, CachedResult>,
    table_hits: u32,
    progress: Vec<(f64, f64)>,
}

impl SolverState {
    fn render_path(&self, matrix: &Matrix) -> String {
        matrix.words(&self.path).join(".")
    }

    fn push_progress(&mut self, k: usize, n: usize) {
        self.progress.push((k as f64 / n as f64, 1.0 / n as f64));
    }

    fn pop_progress(&mut self) {
        self.progress.pop();
    }

    fn render_progress(&mut self, matrix: &Matrix) -> (f64, String) {
        (tree_progress(&self.progress), self.render_path(matrix))
    }
}

struct SolverContext<'a> {
    state: SolverState,
    conf: &'a Solver,
    matrix: &'a Matrix,
    progress_tracker: ProgressTracker<'a>,
}

impl<'a> SolverContext<'a> {
    fn evaluate(
        &mut self,
        current_query: WordId,
        queries: &[WordId],
        outcomes: &Outcomes,
        depth_limit: u8,
    ) -> (Score, Vec<(Response, Option<Rc<Strategy>>)>) {
        let mut total_score = Score::default();
        let mut strategies = Vec::default();
        let num_outcomes = outcomes.len();
        for (j, (response, candidates)) in outcomes.iter().enumerate() {
            self.state.push_progress(j, num_outcomes);
            let min_remaining_depth_sum = (num_outcomes).saturating_sub(j + 2) as u32;
            let (score, strategy) = if candidates.len() == 1 && candidates[0] == current_query {
                (Score::WIN, None)
            } else {
                let (score, strategy) = self.solve(queries, candidates, depth_limit);
                (score, strategy)
            };
            total_score.add_outcome(score);
            self.state.pop_progress();
            if score.depth_max > depth_limit {
                return (Score::LOSS, Default::default());
            }
            strategies.push((*response, strategy));
        }
        (total_score, strategies)
    }

    fn evaluate_many(
        &mut self,
        query_outcomes: &[(WordId, f64, Outcomes)],
        queries: &[WordId],
        num_candidates: usize,
        depth_limit: u8,
    ) -> (Score, Option<Rc<Strategy>>) {
        let mut depth_limit = depth_limit - 1;
        let mut best_strategy = None;
        let mut best_score = Score::LOSS;
        for (i, (query, _, outcomes)) in query_outcomes.iter().enumerate() {
            self.state.push_progress(i, query_outcomes.len());
            let query = *query;
            self.state.path.push(query);
            let (score, strategies) = self.evaluate(query, &queries, outcomes, depth_limit);
            if best_score.ord() < score.ord() {
                best_score = score;
                best_strategy = Some(Rc::new(Strategy {
                    query,
                    outcomes: strategies,
                }));
            }
            depth_limit = best_score.depth_max;
            self.state.path.pop();
            self.state.pop_progress();
        }
        (best_score.ascend(), best_strategy)
    }

    fn solve(
        &mut self,
        queries: &[WordId],
        candidates: &[WordId],
        depth_limit: u8,
    ) -> (Score, Option<Rc<Strategy>>) {
        let num_candidates = candidates.len();
        self.progress_tracker
            .tick(&mut || self.state.render_progress(self.matrix));
        assert_ne!(num_candidates, 0);
        if depth_limit == 0 {
            return (Score::LOSS, None);
        }
        if candidates.len() == 1 {
            let candidate = candidates[0];
            return (
                Score::WIN.ascend(),
                Some(Rc::new(Strategy {
                    query: candidate,
                    outcomes: vec![(self.matrix.response(candidate, candidate), None)],
                })),
            );
        }
        match self.state.table.get(candidates) {
            Some(&CachedResult::DepthExceeds(d)) if d >= depth_limit => {
                return {
                    self.state.table_hits += 1;
                    (Score::LOSS, None)
                }
            }
            Some(&CachedResult::Solved {
                score,
                ref strategy,
            }) => {
                return {
                    self.state.table_hits += 1;
                    (score, strategy.clone())
                }
            }
            _ => {}
        }
        let (queries, query_outcomes) = rank_queries(self.matrix, queries, candidates);
        let &(_, best_k_avg, _) = query_outcomes.first().unwrap();
        let truncated_query_outcomes = &query_outcomes[..query_outcomes
            .iter()
            .position(|&(_, k_avg, _)| k_avg > best_k_avg * (1.0 + self.conf.dk_trunc))
            .unwrap_or(query_outcomes.len())];
        let (score, strategy) = self.evaluate_many(
            truncated_query_outcomes,
            &queries,
            num_candidates,
            depth_limit,
        );
        self.state.table.insert(
            candidates.to_owned(),
            if score.depth_max > depth_limit {
                CachedResult::DepthExceeds(depth_limit)
            } else {
                CachedResult::Solved {
                    score,
                    strategy: strategy.clone(),
                }
            },
        );
        (score, strategy)
    }
}

#[derive(Clone, Debug, Default)]
struct Stats {
    counts: Vec<usize>,
}

impl Stats {
    fn add_sample(&mut self, i: usize) {
        self.counts
            .resize(cmp::max(self.counts.len(), i + 1), Default::default());
        self.counts[i] += 1;
    }

    fn average(&self) -> f64 {
        self.counts
            .iter()
            .enumerate()
            .map(|(i, &n)| i as f64 * n as f64)
            .sum::<f64>()
            / self.counts.iter().map(|&n| n as f64).sum::<f64>()
    }
}

pub fn update_stderr_progress(progress: f64, message: String) {
    if progress == 1.0 {
        eprint!("\x1b[2K\r");
    } else {
        eprint!("\x1b[2K\r{:6.2}% {}", progress * 100.0, message);
    }
}

impl Solver {
    // fn dump_strategy_with<'a>(
    //     &self,
    //     matrix: &Matrix,
    //     queries: &[WordId],
    //     root_queries: &[WordId],
    //     candidates: &[WordId],
    //     depth: usize,
    //     depth_stats: &mut Stats,
    //     mut min_expected_score: i32,
    //     node_counter: usize,
    //     counter: &mut usize,
    //     sink: &mut dyn Write,
    // ) -> io::Result<()> {
    //     let t0 = now();
    //     let strategy = self.solve(
    //         matrix,
    //         queries,
    //         root_queries,
    //         candidates,
    //         &mut update_stderr_progress,
    //     );
    //     if depth == 0 {
    //         eprintln!(
    //             "\nSolve time = {:?}, query = {:?}, score = {}",
    //             now() - t0,
    //             matrix.word(strategy.query),
    //             strategy.score,
    //         );
    //         strategy.score;
    //     }
    //     assert_eq!(strategy.score == 0, candidates.len() <= 1);
    //     if min_expected_score > strategy.score {
    //         eprintln!(
    //             "BUG: expected at least score {}, got {}",
    //             min_expected_score, strategy.score,
    //         );
    //     }
    //     min_expected_score = strategy.score;
    //     if strategy.score == 0 {
    //         depth_stats.add(depth);
    //         return Ok(());
    //     }
    //     let mut outcomes = BTreeMap::default();
    //     for &candidate in candidates {
    //         let response = matrix.response(strategy.query, candidate);
    //         outcomes
    //             .entry(format!("{}", response))
    //             .or_insert(Vec::default())
    //             .push(candidate);
    //     }
    //     let base_counter = *counter;
    //     *counter = base_counter + outcomes.len();
    //     writeln!(
    //         sink,
    //         "# query = {}, depth = {}, score = {}",
    //         matrix.word(strategy.query),
    //         depth,
    //         strategy.score,
    //     )?;
    //     for (i, (response, remaining_candidates)) in outcomes.iter().enumerate() {
    //         writeln!(
    //             sink,
    //             "[{}]\t{}\t{}\t{}\t{}",
    //             node_counter,
    //             matrix.word(strategy.query),
    //             response,
    //             if remaining_candidates.len() > 1 {
    //                 format!("[{}]", base_counter + i)
    //             } else {
    //                 format!("")
    //             },
    //             if remaining_candidates.len() > 1 {
    //                 format!("N={}", remaining_candidates.len())
    //             } else {
    //                 format!("{{{}}}", matrix.words(remaining_candidates).join(", "))
    //             },
    //         )?;
    //     }
    //     for (i, remaining_candidates) in outcomes.values().enumerate() {
    //         self.dump_strategy_with(
    //             matrix,
    //             queries,
    //             queries,
    //             &remaining_candidates,
    //             depth + 1,
    //             depth_stats,
    //             min_expected_score + 1,
    //             base_counter + i,
    //             counter,
    //             sink,
    //         )?;
    //     }
    //     Ok(())
    // }

    // pub fn dump_strategy(
    //     &self,
    //     matrix: &Matrix,
    //     queries: &[WordId],
    //     root_queries: &[WordId],
    //     candidates: &[WordId],
    //     sink: &mut dyn Write,
    // ) -> io::Result<()> {
    //     let mut depth_stats = Default::default();
    //     self.dump_strategy_with(
    //         matrix,
    //         queries,
    //         root_queries,
    //         candidates,
    //         0,
    //         &mut depth_stats,
    //         i32::MIN,
    //         0,
    //         &mut 1,
    //         sink,
    //     )?;
    //     depth_stats.log_summary(&StderrLog);
    //     Ok(())
    // }
}

#[derive(Serialize)]
struct Outcome {
    response: String,
    next_decision: Option<Decision>,
}

#[derive(Serialize)]
struct Decision {
    query: String,
    outcomes: Vec<Outcome>,
}

impl Strategy {
    fn to_decision_tree(&self, matrix: &Matrix) -> Decision {
        Decision {
            query: matrix.word(self.query).into(),
            outcomes: self
                .outcomes
                .iter()
                .map(|(response, next_strategy)| Outcome {
                    response: format!("{}", response),
                    next_decision: next_strategy
                        .as_ref()
                        .map(|strategy| strategy.to_decision_tree(matrix)),
                })
                .collect(),
        }
    }
}

impl Decision {
    fn accumulate_depth_stats(&self, depth: usize, stats: &mut Stats) {
        for Outcome { next_decision, .. } in &self.outcomes {
            match next_decision {
                None => stats.add_sample(depth),
                Some(decision) => decision.accumulate_depth_stats(depth + 1, stats),
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn solve(words: &str, guesses: &str, num_roots: u32, dk_trunc: f64) -> io::Result<()> {
    if words.is_empty() {
        return Err(error("Please upload a word list first."));
    }
    let mut database = Database::parse(words)?;
    let guesses = parse_guesses(guesses, "\n")?;
    database.candidates = filter_candidates(&guesses, &database.candidates);

    let mut candidates = database.candidates.clone();
    if candidates.is_empty() {
        return Err(error("no candidates remain"));
    }
    candidates.sort();
    js::Reply::SetCandidates { candidates }.post();

    let matrix = {
        let _timer = js::Timer::from("preprocess");
        Matrix::build(&database, &mut |progress, message| {
            js::update_progress(format!("(0/2) Preprocessing... {}", message), progress);
        })?
    };

    let candidates: Vec<WordId> = (0..database.candidates.len())
        .map(|c| WordId(c.try_into().unwrap()))
        .collect();
    let queries: Vec<WordId> = (0..matrix.words.len()).map(From::from).collect();

    let solver = Solver { dk_trunc };
    js::update_progress(format!("(1/2) Solving..."), 0.0);
    {
        let _timer = js::Timer::from("solve");
        let (score, strategy) =
            solver.solve(&matrix, &queries, &candidates, &mut |progress, message| {
                js::update_progress(format!("(1/2) Solving... {}", message), progress);
            });
        if let Some(strategy) = strategy {
            let decision_tree = strategy.to_decision_tree(&matrix);
            let mut depths = Stats::default();
            decision_tree.accumulate_depth_stats(1, &mut depths);
            js::Reply::ReportStrategy {
                depths: depths.counts.clone(),
                depth_avg: depths.average(),
                decision_tree,
            }
            .post();
        }
    }
    js::update_status(format!("(2/2) Done!"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_string() {
        for response in ["22222", "20220", "02022", "12010", "11210", "00000"] {
            assert_eq!(
                format!("{}", Response::try_from(response).unwrap()),
                response,
            );
        }
    }

    #[test]
    fn response_compute() {
        for (query, candidate, response) in [
            ("abcde", "abcde", "22222"),
            ("apcdq", "abcde", "20220"),
            ("pbqde", "abcde", "02022"),
            ("bbbde", "abcbd", "12010"),
            ("bbcdc", "adcbb", "11210"),
            ("pqrst", "abcbd", "00000"),
        ] {
            assert_eq!(
                Response::compute(query, candidate),
                Response::try_from(response).unwrap(),
                "{} {}",
                query,
                candidate,
            );
        }
    }
}
