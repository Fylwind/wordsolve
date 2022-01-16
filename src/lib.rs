#![crate_type = "cdylib"]
#![crate_type = "lib"]

use float_ord::FloatOrd;
use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use lazy_static::lazy_static;
use static_assertions::const_assert;
use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::io::Write;
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
        let query = word_chars(query).unwrap();
        let solution = word_chars(solution).unwrap();
        Self::compute_with(query, solution)
    }

    fn compute_with(query: [WordChar; NUM_CHARS], mut solution: [WordChar; NUM_CHARS]) -> Self {
        let mut color_buf = [Color::GRAY; NUM_CHARS];
        for ((color, &query_char), solution_char) in
            color_buf.iter_mut().zip(&query).zip(&mut solution)
        {
            if query_char == *solution_char {
                *color = Color::GREEN;
                *solution_char = WordChar::INVALID;
            }
        }
        for (color, &query_char) in color_buf.iter_mut().zip(&query) {
            if *color == Color::GRAY {
                for solution_char in &mut solution {
                    if query_char == *solution_char {
                        *color = Color::YELLOW;
                        *solution_char = WordChar::INVALID;
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
    pub fn build(database: &Database, progress_sink: &mut ProgressSink) -> io::Result<Self> {
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
        let queries: Vec<[WordChar; NUM_CHARS]> = words
            .iter()
            .map(|word| word_chars(word))
            .collect::<io::Result<Vec<_>>>()?;
        let solutions: Vec<[WordChar; NUM_CHARS]> = database
            .solutions
            .iter()
            .map(|word| word_chars(word))
            .collect::<io::Result<Vec<_>>>()?;
        let mut progress_tracker = ProgressTracker::new(progress_sink);
        for (i, &query) in queries.iter().enumerate() {
            progress_tracker
                .tick(&mut || (i as f64 / queries.len() as f64, format!("{}", words[i])));
            for &solution in &solutions {
                let response = Response::compute_with(query, solution);
                responses.push(response)
            }
        }
        Ok(Self {
            responses,
            num_solutions: database.solutions.len(),
            words,
            word_ids,
        })
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
    pub fn parse(s: &str) -> io::Result<Self> {
        let mut solutions = Vec::default();
        let mut nonsolutions = Vec::default();
        for line in s.lines() {
            let word = line.trim().to_lowercase();
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
    render_frequency: u32,
    t_start: f64,
}

impl<'a> ProgressTracker<'a> {
    const RENDER_INTERVAL: time::Duration = time::Duration::from_millis(50);

    fn new(progress_sink: &'a mut ProgressSink<'a>) -> Self {
        Self {
            progress_sink,
            ticks_before_render: 1,
            ticks_total: 1,
            render_frequency: 1,
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
        self.render_frequency = ((Self::RENDER_INTERVAL.as_secs_f64() / elapsed
            * self.ticks_total as f64) as u32)
            .clamp(self.render_frequency / 2, self.render_frequency * 2)
            .clamp(1, 1000000);
        self.ticks_before_render = self.render_frequency;
        self.ticks_total += self.render_frequency as u64;
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

impl Solver {
    fn solve(
        &self,
        matrix: &Matrix,
        queries: &[WordId],
        root_queries: &[WordId],
        candidates: &[WordId],
        progress_sink: &mut ProgressSink,
    ) -> Strategy {
        let mut state = SolverState::default();
        state.cache = vec![Default::default(); self.max_depth as usize];
        state.root_queries = root_queries.iter().copied().collect();
        let mut context = SolverContext {
            state,
            conf: self,
            matrix,
            progress_tracker: ProgressTracker::new(progress_sink),
        };
        context.solve(queries, candidates, 0, i32::MIN, i32::MAX)
    }
}

#[derive(Clone, Debug, Default)]
struct SolverState {
    path: Vec<WordId>,
    map_buf: HashMap<Response, Vec<WordId>>,
    cache: Vec<HashMap<Vec<WordId>, Strategy>>,
    cache_hits: u32,
    /// Restricts the set of queries at the root.
    /// Does not affect its children.
    root_queries: HashSet<WordId>,
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

    fn render_progress(&mut self, matrix: &Matrix, num_candidates: usize) -> (f64, String) {
        (
            tree_progress(&self.progress),
            format!(
                "[words={} nc={} hit={}/{}]",
                self.render_path(matrix),
                num_candidates,
                self.cache_hits,
                self.cache.iter().map(|x| x.len()).sum::<usize>(),
            ),
        )
    }
}

struct SolverContext<'a> {
    state: SolverState,
    conf: &'a Solver,
    matrix: &'a Matrix,
    progress_tracker: ProgressTracker<'a>,
}

impl<'a> SolverContext<'a> {
    fn solve(
        &mut self,
        queries: &[WordId],
        candidates: &[WordId],
        depth: i32,
        alpha: i32,
        beta: i32,
    ) -> Strategy {
        let num_candidates = candidates.len();
        self.progress_tracker
            .tick(&mut || self.state.render_progress(self.matrix, num_candidates));
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
        if let Some(&strategy) = self.state.cache[depth as usize].get(candidates) {
            self.state.cache_hits += 1;
            return strategy;
        }
        let mut query_outcomes = Vec::default();
        for query in queries {
            let query = *query;
            if let Some((entropy, outcomes)) =
                divide_candidates(self.matrix, query, candidates, &mut self.state.map_buf)
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
                .filter(|(query, _, _)| self.state.root_queries.contains(query))
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
            self.state.push_progress(
                i,
                cmp::min(self.conf.max_branching as usize, query_outcomes.len()),
            );
            let query = *query;
            self.state.path.push(query);
            let mut worst_score = i32::MAX;
            {
                let mut beta = beta;
                for (j, (_, remaining_candidates)) in outcomes.iter().enumerate() {
                    self.state.push_progress(j, outcomes.len());
                    let mut strategy = self.solve(
                        &effective_queries,
                        remaining_candidates,
                        depth + 1,
                        alpha,
                        beta,
                    );
                    if remaining_candidates.len() == 1 && remaining_candidates[0] == query {
                        strategy.score -= 1;
                    }
                    worst_score = cmp::min(worst_score, strategy.score);
                    beta = cmp::min(beta, strategy.score);
                    self.state.pop_progress();
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
            self.state.path.pop();
            self.state.pop_progress();
            if beta <= alpha {
                break;
            }
        }
        self.state.cache[depth as usize].insert(candidates.to_owned(), best_strategy);
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

impl Solver {
    fn dump_strategy_with<'a>(
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
        log: &'a dyn Log,
    ) -> io::Result<()> {
        let t0 = now();
        let strategy = self.solve(
            matrix,
            queries,
            root_queries,
            candidates,
            &mut |progress, message| {
                log.log(&format!("\x1b[2K\r{:6.3}%\t{}", progress, message));
            },
        );
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

#[cfg(target_arch = "wasm32")]
fn solve(words: &str, guesses: &str, max_branching: i32, log: &dyn Log) -> io::Result<()> {
    if words.is_empty() {
        return Err(error("Please upload a word list first."));
    }
    let mut database = Database::parse(words)?;
    let guesses = parse_guesses(guesses, "\n")?;
    database.solutions = filter_candidates(&guesses, &database.solutions);

    let mut candidates = database.solutions.clone();
    if candidates.is_empty() {
        return Err(error("no candidates remain"));
    }
    candidates.sort();
    js::OutMessage::SetCandidates { candidates }.post();

    let matrix = {
        let _timer = js::Timer::from("preprocess");
        Matrix::build(&database, &mut |progress, message| {
            js::update_progress(format!("(0/2) Preprocessing... {}", message), progress);
        })?
    };

    let candidates: Vec<WordId> = (0..database.solutions.len())
        .map(|c| WordId(c.try_into().unwrap()))
        .collect();
    let queries: Vec<WordId> = (0..matrix.words.len()).map(From::from).collect();

    let solver = Solver {
        max_depth: 9,
        max_branching: max_branching,
    };
    js::update_progress(format!("(1/2) Solving..."), 0.0);
    {
        let _timer = js::Timer::from("solve");
        let strategy = solver.solve(
            &matrix,
            &queries,
            &queries,
            &candidates,
            &mut |progress, message| {
                js::update_progress(format!("(1/2) Solving... {}", message), progress);
            },
        );
        log.log(&format!("{:?}", strategy));
        js::OutMessage::AppendQuery {
            query: vec![
                matrix.word(strategy.query).into(),
                format!("{}", strategy.score),
            ],
        }
        .post();
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
