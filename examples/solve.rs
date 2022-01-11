use clap::Parser;
use float_ord::FloatOrd;
use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use serde::Deserialize;
use static_assertions::const_assert;
use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::io::Write;
use std::path::PathBuf;
use std::{cmp, fmt, fs, io, str, time};

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
struct Response(u8);

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
            return Err(error("length mismatch"));
        }
        const_assert!(Color::COUNT as u32 as usize == Color::COUNT);
        s.chars()
            .map(|c| {
                let digit = c
                    .to_digit(Color::COUNT as u32)
                    .ok_or(error("invalid digit"))?;
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
struct WordId(u16);

impl From<usize> for WordId {
    fn from(i: usize) -> Self {
        Self(i.try_into().expect("WordId overflow"))
    }
}

#[derive(Clone, Debug)]
struct Matrix {
    responses: Vec<Response>,
    num_solutions: usize,
    words: Vec<String>,
    word_ids: HashMap<String, WordId>,
}

impl Matrix {
    fn build(database: &Database) -> Self {
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

    #[allow(dead_code)]
    fn word_ids(&self, words: &[String]) -> Vec<WordId> {
        words
            .iter()
            .map(|word| *self.word_ids.get(word).expect("word not found"))
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize)]
struct Database {
    /// Words that are valid as a solution.
    solutions: Vec<String>,
    /// Words that are valid as input, but never as a solution.
    nonsolutions: Vec<String>,
}

fn parse_word_response(s: &str) -> io::Result<(&str, Response)> {
    let fields: Vec<&str> = s.trim().split_whitespace().collect();
    if fields.len() != 2 {
        return Err(error("wrong number of fields"));
    }
    Ok((fields[0], Response::try_from(fields[1])?))
}

fn filter_candidates(word: &str, response: Response, candidates: &[String]) -> Vec<String> {
    candidates
        .iter()
        .cloned()
        .filter(|candidate| Response::compute(word, candidate) == response)
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Strategy {
    query: WordId,
    score: i32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Solver {
    max_branching: i32,
    max_depth: i32,
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
    t_start: Option<time::Instant>,
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

impl SolverState {
    fn render_path(&self, matrix: &Matrix) -> String {
        matrix.words(&self.path).join(".")
    }

    fn render_progress(&mut self, matrix: &Matrix, num_candidates: usize) {
        const RENDER_INTERVAL: time::Duration = time::Duration::from_millis(200);
        let progress = self
            .progress
            .iter()
            .rfold(0.0, |b, &(k, n)| (k as f64 + b) / n as f64);
        let elapsed = self.t_start.expect("missing t_start").elapsed();
        let remaining = elapsed.as_secs_f64() * (1.0 - progress) / progress;
        eprint!(
            "\x1b[2K\r{:6.3}%\t{}\t# ETA={} hit={:07}/{:07} cand={:04} {}",
            progress * 100.0,
            render_duration_secs(elapsed.as_secs_f64()),
            render_duration_secs(remaining),
            self.cache_hits,
            self.cache.iter().map(|x| x.len()).sum::<usize>(),
            num_candidates,
            self.render_path(matrix),
        );
        io::stderr().flush().expect("flush failed");
        self.ticks_before_render = ((RENDER_INTERVAL.as_secs_f64() / elapsed.as_secs_f64()
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
    ) -> Strategy {
        let num_candidates = candidates.len();
        self.ticks_before_render -= 1;
        if self.ticks_before_render == 0 {
            self.render_progress(matrix, num_candidates);
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
            let outcomes = &mut self.map_buf;
            outcomes.clear();
            for &candidate in candidates {
                let response = matrix.response(query, candidate);
                let candidates = outcomes.entry(response).or_insert(Vec::default());
                candidates.push(candidate);
            }
            if outcomes.len() > 1 {
                let mut entropy = 0.0;
                let mut outcomes: Vec<_> = outcomes.drain().collect();
                outcomes.sort_unstable_by_key(|(response, candidates)| {
                    (cmp::Reverse(candidates.len()), *response)
                });
                for (_, candidates) in &outcomes {
                    let p = candidates.len() as f64 / num_candidates as f64;
                    entropy += -p * f64::log2(p);
                }
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
                    let strategy = self.solve(
                        matrix,
                        &effective_queries,
                        remaining_candidates,
                        depth + 1,
                        alpha,
                        beta,
                    );
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

impl DepthStats {
    fn add(&mut self, depth: usize) {
        self.counts
            .resize(cmp::max(self.counts.len(), depth + 1), Default::default());
        self.counts[depth] += 1;
    }

    fn eprint_summary(&self) {
        let n: usize = self.counts.iter().copied().sum();
        eprintln!("depth distribution:");
        for (i, &k) in self.counts.iter().enumerate() {
            eprintln!("\t{}\t{:5.1}%", i, k as f64 / n as f64 * 100.0);
        }
    }
}

impl Solver {
    fn solve(
        &self,
        matrix: &Matrix,
        queries: &[WordId],
        root_queries: &[WordId],
        candidates: &[WordId],
    ) -> Strategy {
        let mut state = SolverState::default();
        state.conf = *self;
        state.cache = vec![Default::default(); self.max_depth as usize];
        state.root_queries = root_queries.iter().copied().collect();
        state.t_start = Some(time::Instant::now());
        state.ticks_before_render = 1000;
        state.ticks_total = state.ticks_before_render as u64;
        let strategy = state.solve(matrix, queries, candidates, 0, i32::MIN, i32::MAX);
        eprint!("\x1b[2K\r");
        io::stderr().flush().unwrap();
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
    ) -> io::Result<()> {
        let t0 = time::Instant::now();
        let strategy = self.solve(matrix, queries, root_queries, candidates);
        if depth == 0 {
            eprintln!(
                "\nSolve time = {:?}, query = {:?}, score = {}",
                t0.elapsed(),
                matrix.word(strategy.query),
                strategy.score,
            );
            strategy.score;
        }
        assert_eq!(strategy.score == 0, candidates.len() <= 1);
        if min_expected_score > strategy.score {
            eprintln!(
                "BUG: expected at least score {}, got {}",
                min_expected_score, strategy.score,
            );
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
            )?;
        }
        Ok(())
    }

    fn dump_strategy(
        &self,
        matrix: &Matrix,
        queries: &[WordId],
        root_queries: &[WordId],
        candidates: &[WordId],
        sink: &mut dyn Write,
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
        )?;
        depth_stats.eprint_summary();
        Ok(())
    }
}

#[derive(Parser, Clone, Debug)]
struct Args {
    #[clap(long, short = 'd', help = "word database in JSON")]
    database: PathBuf,
    #[clap(long, short = 'o')]
    out_strategy: PathBuf,
    #[clap(long)]
    exclude_nonsolutions: bool,
    /// Filter candidates by responses. Example: -f 'hello 01001; world 02200'
    #[clap(long, short = 'f', default_value = "")]
    filters: String,
    #[clap(long, default_value = "100")]
    max_branching: i32,
    #[clap(long, default_value = "10")]
    max_depth: i32,
    #[clap(long)]
    root_queries: Vec<String>,
    #[clap(long)]
    truncate_solutions: Option<usize>,
}

pub fn main() -> io::Result<()> {
    let args = Args::parse();

    let file = io::BufReader::new(fs::File::open(args.database)?);
    let mut database: Database = serde_json::de::from_reader(file)?;
    for entry in args.filters.split(";") {
        if entry.trim().is_empty() {
            continue;
        }
        let (query, response) = parse_word_response(&entry)?;
        database.solutions = filter_candidates(query, response, &database.solutions);
    }
    if let Some(limit) = args.truncate_solutions {
        assert!(limit <= database.solutions.len());
        database.solutions.resize(limit, Default::default());
    }
    if args.exclude_nonsolutions {
        database.nonsolutions.clear();
    }
    if database.solutions.is_empty() {
        return Err(error("no solution candidates remain"));
    }
    database.solutions.sort();
    println!(
        "{} {} {}",
        database.solutions.len(),
        database.nonsolutions.len(),
        database.solutions.len() + database.nonsolutions.len()
    );
    database.nonsolutions.sort();

    eprintln!("Building matrix...");
    let t0 = time::Instant::now();
    let matrix = Matrix::build(&database);
    eprintln!("time = {:?}", t0.elapsed());
    let candidates: Vec<WordId> = (0..database.solutions.len())
        .map(|c| WordId(c.try_into().unwrap()))
        .collect();
    let queries: Vec<WordId> = (0..matrix.words.len()).map(From::from).collect();
    let root_queries = matrix.word_ids(&args.root_queries);
    let root_queries = if root_queries.is_empty() {
        &queries
    } else {
        &root_queries
    };

    let solver = Solver {
        max_depth: args.max_depth,
        max_branching: args.max_branching,
    };
    let mut out_file = io::BufWriter::new(fs::File::create(&args.out_strategy)?);
    eprintln!("Solving...");
    solver.dump_strategy(&matrix, &queries, root_queries, &candidates, &mut out_file)?;

    Ok(())
}
