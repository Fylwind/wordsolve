#![crate_type = "cdylib"]
#![crate_type = "lib"]

use float_ord::FloatOrd;
use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use lazy_static::lazy_static;
use static_assertions::const_assert;
use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::io::{BufRead, Write};
use std::{cmp, fmt, io, str, time};

#[cfg(target_arch = "wasm32")]
mod js;

#[cfg(target_arch = "wasm32")]
fn solve(words: &[String], guesses: &str, log: &dyn Log) -> io::Result<()> {
    if words.len() == 0 {
        return Err(error("Error: Please upload a word list first."));
    }
    let nonsolutions = words
        .iter()
        .filter(|w| w.starts_with("!"))
        .map(|w| w.strip_prefix('!').unwrap_or(w).to_owned())
        .collect();
    let candidates = words
        .iter()
        .filter(|w| !w.starts_with("!"))
        .cloned()
        .collect();
    let mut database = Database {
        solutions: candidates,
        nonsolutions: nonsolutions,
    };
    js::OutMessage::ClearLog.post();
    log.log("Preprocessing...");
    let guesses = parse_guesses(guesses, "\n")?;
    database.solutions = filter_candidates(&guesses, &database.solutions);

    let mut candidates = database.solutions.clone();
    if candidates.is_empty() {
        return Err(error("no candidates remain"));
    }
    candidates.sort();
    js::OutMessage::SetCandidates { candidates }.post();

    let matrix = Matrix::build(&database);

    let candidates: Vec<WordId> = (0..database.solutions.len())
        .map(|c| WordId(c.try_into().unwrap()))
        .collect();
    let queries: Vec<WordId> = (0..matrix.words.len()).map(From::from).collect();

    let solver = Solver {
        max_depth: 9,
        max_branching: 1,
    };
    let mut out_file = io::sink();
    log.log("Solving...");
    solver.dump_strategy(&matrix, &queries, &queries, &candidates, &mut out_file, log)
}

fn error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Other, message)
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

const NUM_CHARS: usize = 5;

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Response(u8);

impl Response {
    const MAX: Self = Self(usize::pow(Color::COUNT, NUM_CHARS as u32) as u8);
}

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
    fn compute(query: &str, solution: &str) -> Self {
        let query: Vec<char> = query.chars().collect();
        let solution: Vec<char> = solution.chars().collect();
        Self::compute_with(
            &query,
            &solution,
            &mut Default::default(),
            &mut Default::default(),
        )
    }

    fn compute_with(
        query: &[char],
        solution: &[char],
        color_buf: &mut Vec<Color>,
        mask_buf: &mut Vec<bool>,
    ) -> Self {
        let n = NUM_CHARS;
        assert_eq!(n, query.len());
        assert_eq!(n, solution.len());
        color_buf.clear();
        color_buf.resize(n, Color::GRAY);
        mask_buf.clear();
        mask_buf.resize(n, false);
        for (((color, &query_char), mask), &solution_char) in color_buf
            .iter_mut()
            .zip(query)
            .zip(mask_buf.iter_mut())
            .zip(solution)
        {
            if query_char == solution_char {
                *color = Color::GREEN;
                *mask = true;
            }
        }
        for (color, &query_char) in color_buf.iter_mut().zip(query) {
            if *color == Color::GRAY {
                for (mask, &solution_char) in mask_buf.iter_mut().zip(solution) {
                    if !*mask && query_char == solution_char {
                        *color = Color::YELLOW;
                        *mask = true;
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
    num_solutions: usize,
    pub words: Vec<String>,
    pub word_ids: HashMap<String, WordId>,
}

impl Matrix {
    pub fn build(database: &Database) -> Self {
        let mut responses = Vec::default();
        let words: Vec<String> = database
            .solutions
            .iter()
            .chain(&database.nonsolutions)
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
        let queries: Vec<Vec<char>> = words.iter().map(|word| word.chars().collect()).collect();
        let solutions: Vec<Vec<char>> = database
            .solutions
            .iter()
            .map(|word| word.chars().collect())
            .collect();
        let mut color_buf = Default::default();
        let mut mask_buf = Default::default();
        for query in &queries {
            for solution in &solutions {
                let response =
                    Response::compute_with(query, solution, &mut color_buf, &mut mask_buf);
                responses.push(response)
            }
        }
        Self {
            responses,
            num_solutions: database.solutions.len(),
            words,
            word_ids,
        }
    }

    fn response(&self, query: WordId, solution: WordId) -> Response {
        assert!((solution.0 as usize) < self.num_solutions);
        self.responses[query.0 as usize * self.num_solutions + solution.0 as usize]
    }

    fn word(&self, WordId(i): WordId) -> &str {
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
    pub solutions: Vec<String>,
    /// Words that are valid as input, but never as a solution.
    pub nonsolutions: Vec<String>,
}

impl Database {
    pub fn read(r: &mut dyn BufRead) -> io::Result<Self> {
        let mut solutions = Vec::default();
        let mut nonsolutions = Vec::default();
        for line in r.lines() {
            let word = line?.trim().to_lowercase();
            let is_nonsolution = word.starts_with("!");
            let word = word.strip_prefix('!').unwrap_or(&word);
            let chars: Vec<char> = word.chars().collect();
            if !(chars.len() == NUM_CHARS && chars.iter().all(char::is_ascii_lowercase)) {
                continue;
            }
            if is_nonsolution {
                &mut nonsolutions
            } else {
                &mut solutions
            }
            .push(word.to_owned());
        }
        Ok(Database {
            solutions,
            nonsolutions,
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

#[derive(Clone, Copy, Debug)]
struct Strategy {
    query: WordId,
    score: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Solver {
    pub max_branching: i32,
    pub max_depth: i32,
}

#[derive(Clone, Debug, Default)]
struct SolverState {
    conf: Solver,
    path: Vec<WordId>,
    map_buf: HashMap<Response, Vec<WordId>>,
    cache: Vec<HashMap<Vec<WordId>, Strategy>>,
    cache_hits: u32,
    /// Restricts the set of queries at the root.
    /// Does not affect its children.
    root_queries: HashSet<WordId>,
    progress: Vec<(usize, usize)>,
    ticks_before_render: u32,
    ticks_total: u64,
    t_start: Option<f64>,
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
    if secs >= 864000.0 {
        format!("?:??:??:??")
    } else {
        format!(
            "{:01}:{:02}:{:02}:{:02}",
            f64::floor(secs / 86400.0),
            f64::floor(secs / 3600.0 % 60.0),
            f64::floor(secs / 60.0 % 60.0),
            f64::floor(secs % 60.0),
        )
    }
}

fn divide_candidates(
    matrix: &Matrix,
    query: WordId,
    candidates: &[WordId],
    outcomes_buf: &mut HashMap<Response, Vec<WordId>>,
) -> Option<(f64, Vec<(Response, Vec<WordId>)>)> {
    outcomes_buf.clear();
    for &candidate in candidates {
        let response = matrix.response(query, candidate);
        let candidates = outcomes_buf.entry(response).or_insert(Vec::default());
        candidates.push(candidate);
    }
    if outcomes_buf.len() <= 1 {
        return None;
    }
    let mut entropy = 0.0;
    let mut outcomes: Vec<_> = outcomes_buf.drain().collect();
    outcomes
        .sort_unstable_by_key(|(response, candidates)| (cmp::Reverse(candidates.len()), *response));
    let n = candidates.len();
    for (_, candidates) in &outcomes {
        let p = candidates.len() as f64 / n as f64;
        entropy += -p * f64::log2(p);
    }
    Some((entropy, outcomes))
}

impl SolverState {
    fn render_path(&self, matrix: &Matrix) -> String {
        matrix.words(&self.path).join(".")
    }

    fn render_progress(&mut self, matrix: &Matrix, num_candidates: usize, log: &dyn Log) {
        const RENDER_INTERVAL: time::Duration = time::Duration::from_millis(200);
        let progress = self
            .progress
            .iter()
            .rfold(0.0, |b, &(k, n)| (k as f64 + b) / n as f64);
        let elapsed = now() - self.t_start.expect("missing t_start");
        let remaining = elapsed * (1.0 - progress) / progress;
        log.log(&format!(
            "\x1b[2K\r{:6.3}%\t{}\t# ETA={} hit={:07}/{:07} cand={:04} {}",
            progress * 100.0,
            render_duration_secs(elapsed),
            render_duration_secs(remaining),
            self.cache_hits,
            self.cache.iter().map(|x| x.len()).sum::<usize>(),
            num_candidates,
            self.render_path(matrix),
        ));
        self.ticks_before_render = ((RENDER_INTERVAL.as_secs_f64() / elapsed
            * self.ticks_total as f64) as u32)
            .clamp(1, 1000000);
        self.ticks_total += self.ticks_before_render as u64;
    }

    fn solve(
        &mut self,
        matrix: &Matrix,
        queries: &[WordId],
        candidates: &[WordId],
        depth: i32,
        alpha: i32,
        beta: i32,
        log: &dyn Log,
    ) -> Strategy {
        let num_candidates = candidates.len();
        self.ticks_before_render -= 1;
        if self.ticks_before_render == 0 {
            self.render_progress(matrix, num_candidates, log);
        }
        let fallback_query = candidates[0];
        if num_candidates <= 1 {
            return Strategy {
                query: fallback_query,
                score: -depth,
            };
        }
        let pessimistic_score = 1 - num_candidates as i32 - depth;
        let optimistic_score = -ceil_log(num_candidates as i32, Response::MAX.0 as i32) - depth;
        if depth >= self.conf.max_depth {
            return Strategy {
                query: fallback_query,
                score: pessimistic_score,
            };
        }
        if optimistic_score <= alpha {
            return Strategy {
                query: fallback_query,
                score: pessimistic_score,
            };
        }
        if pessimistic_score >= beta {
            return Strategy {
                query: fallback_query,
                score: pessimistic_score,
            };
        }
        if let Some(&strategy) = self.cache[depth as usize].get(candidates) {
            self.cache_hits += 1;
            return strategy;
        }
        let mut query_outcomes = Vec::default();
        for query in queries {
            let query = *query;
            if let Some((entropy, outcomes)) =
                divide_candidates(matrix, query, candidates, &mut self.map_buf)
            {
                query_outcomes.push((query, entropy, outcomes));
            }
        }
        let effective_queries: Vec<WordId> =
            query_outcomes.iter().map(|&(query, _, _)| query).collect();
        query_outcomes.sort_by_key(|(_, entropy, outcomes)| {
            (outcomes.first().unwrap().1.len(), FloatOrd(-*entropy))
        });
        if depth == 0 {
            query_outcomes = query_outcomes
                .into_iter()
                .filter(|(query, _, _)| self.root_queries.contains(query))
                .collect();
        }
        let mut best_strategy = Strategy {
            query: fallback_query,
            score: i32::MIN,
        };
        let mut alpha = alpha;
        for (i, (query, _, outcomes)) in query_outcomes
            .iter()
            .take(self.conf.max_branching as usize)
            .enumerate()
        {
            self.progress.push((
                i,
                cmp::min(self.conf.max_branching as usize, query_outcomes.len()),
            ));
            let query = *query;
            self.path.push(query);
            let mut worst_score = i32::MAX;
            {
                let mut beta = beta;
                for (j, (_, remaining_candidates)) in outcomes.iter().enumerate() {
                    self.progress.push((j, outcomes.len()));
                    let mut strategy = self.solve(
                        matrix,
                        &effective_queries,
                        remaining_candidates,
                        depth + 1,
                        alpha,
                        beta,
                        log,
                    );
                    if remaining_candidates.len() == 1 && remaining_candidates[0] == query {
                        strategy.score -= 1;
                    }
                    worst_score = cmp::min(worst_score, strategy.score);
                    beta = cmp::min(beta, strategy.score);
                    self.progress.pop();
                    if beta <= alpha {
                        break;
                    }
                }
            }
            if best_strategy.score < worst_score {
                // IMPORTANT: Because we are short-circuiting via beta <=
                // alpha, this has to be "<", not "<=". After finding an
                // optimal solution, the alpha would prevent full exploration
                // of other trees.
                best_strategy.query = query;
                best_strategy.score = worst_score;
            }
            alpha = cmp::max(alpha, worst_score);
            self.path.pop();
            self.progress.pop();
            if beta <= alpha {
                break;
            }
        }
        self.cache[depth as usize].insert(candidates.to_owned(), best_strategy);
        best_strategy
    }
}

#[derive(Clone, Debug, Default)]
struct DepthStats {
    counts: Vec<usize>,
}

pub trait Log {
    fn log(&self, message: &str);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StderrLog;

impl Log for StderrLog {
    fn log(&self, message: &str) {
        if message.starts_with("\x1b[2K\r") {
            eprint!("{}", message);
            io::stderr().flush().expect("flush failed");
        } else {
            eprintln!("{}", message);
        }
    }
}

impl DepthStats {
    fn add(&mut self, depth: usize) {
        self.counts
            .resize(cmp::max(self.counts.len(), depth + 1), Default::default());
        self.counts[depth] += 1;
    }

    fn log_summary(&self, log: &dyn Log) {
        let n: usize = self.counts.iter().copied().sum();
        log.log("depth distribution:");
        for (i, &k) in self.counts.iter().enumerate() {
            log.log(&format!("\t{}\t{:5.1}%", i, k as f64 / n as f64 * 100.0));
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn now() -> f64 {
    js::now() / 1e6
}

lazy_static! {
    static ref REFERENCE_INSTANT: time::Instant = time::Instant::now();
}

#[cfg(not(target_arch = "wasm32"))]
fn now() -> f64 {
    let reference = *REFERENCE_INSTANT; // This must run first!
    time::Instant::now().duration_since(reference).as_secs_f64()
}

impl Solver {
    fn solve(
        &self,
        matrix: &Matrix,
        queries: &[WordId],
        root_queries: &[WordId],
        candidates: &[WordId],
        log: &dyn Log,
    ) -> Strategy {
        let mut state = SolverState::default();
        state.conf = *self;
        state.cache = vec![Default::default(); self.max_depth as usize];
        state.root_queries = root_queries.iter().copied().collect();
        state.t_start = Some(now());
        state.ticks_before_render = 1000;
        state.ticks_total = state.ticks_before_render as u64;
        let strategy = state.solve(matrix, queries, candidates, 0, i32::MIN, i32::MAX, log);
        log.log("\x1b[2K\r");
        strategy
    }

    fn dump_strategy_with(
        &self,
        matrix: &Matrix,
        queries: &[WordId],
        root_queries: &[WordId],
        candidates: &[WordId],
        depth: usize,
        depth_stats: &mut DepthStats,
        mut min_expected_score: i32,
        node_counter: usize,
        counter: &mut usize,
        sink: &mut dyn Write,
        log: &dyn Log,
    ) -> io::Result<()> {
        let t0 = now();
        let strategy = self.solve(matrix, queries, root_queries, candidates, log);
        if depth == 0 {
            log.log(&format!(
                "\nSolve time = {:?}, query = {:?}, score = {}",
                now() - t0,
                matrix.word(strategy.query),
                strategy.score,
            ));
            strategy.score;
        }
        assert_eq!(strategy.score == 0, candidates.len() <= 1);
        if min_expected_score > strategy.score {
            log.log(&format!(
                "BUG: expected at least score {}, got {}",
                min_expected_score, strategy.score,
            ));
        }
        min_expected_score = strategy.score;
        if strategy.score == 0 {
            depth_stats.add(depth);
            return Ok(());
        }
        let mut outcomes = BTreeMap::default();
        for &candidate in candidates {
            let response = matrix.response(strategy.query, candidate);
            outcomes
                .entry(format!("{}", response))
                .or_insert(Vec::default())
                .push(candidate);
        }
        let base_counter = *counter;
        *counter = base_counter + outcomes.len();
        writeln!(
            sink,
            "# query = {}, depth = {}, score = {}",
            matrix.word(strategy.query),
            depth,
            strategy.score,
        )?;
        for (i, (response, remaining_candidates)) in outcomes.iter().enumerate() {
            writeln!(
                sink,
                "[{}]\t{}\t{}\t{}\t{}",
                node_counter,
                matrix.word(strategy.query),
                response,
                if remaining_candidates.len() > 1 {
                    format!("[{}]", base_counter + i)
                } else {
                    format!("")
                },
                if remaining_candidates.len() > 1 {
                    format!("N={}", remaining_candidates.len())
                } else {
                    format!("{{{}}}", matrix.words(remaining_candidates).join(", "))
                },
            )?;
        }
        for (i, remaining_candidates) in outcomes.values().enumerate() {
            self.dump_strategy_with(
                matrix,
                queries,
                queries,
                &remaining_candidates,
                depth + 1,
                depth_stats,
                min_expected_score + 1,
                base_counter + i,
                counter,
                sink,
                log,
            )?;
        }
        Ok(())
    }

    pub fn dump_strategy(
        &self,
        matrix: &Matrix,
        queries: &[WordId],
        root_queries: &[WordId],
        candidates: &[WordId],
        sink: &mut dyn Write,
        log: &dyn Log,
    ) -> io::Result<()> {
        let mut depth_stats = Default::default();
        self.dump_strategy_with(
            matrix,
            queries,
            root_queries,
            candidates,
            0,
            &mut depth_stats,
            i32::MIN,
            0,
            &mut 1,
            sink,
            log,
        )?;
        depth_stats.log_summary(log);
        Ok(())
    }
}
