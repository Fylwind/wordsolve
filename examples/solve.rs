use clap::Parser;
use std::path::PathBuf;
use std::{fs, io, str, time};
use wordsolve;
use wordsolve::{Database, Matrix, Solver, WordId};

fn error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Other, message)
}

#[derive(Parser, Clone, Debug)]
struct Args {
    #[clap(long, short = 'd', help = "word database in JSON")]
    database: PathBuf,
    // #[clap(long, short = 'o')]
    // out_strategy: PathBuf,
    /// Filter candidates by responses. Example: -f 'hello 01001; world 02200'
    #[clap(long, short = 'f', default_value = "")]
    filters: String,
    #[clap(long, short = 'k', default_value = "1")]
    dk_trunc: f64,
    #[clap(long)]
    truncate_candidates: Option<usize>,
}

pub fn main() -> io::Result<()> {
    let args = Args::parse();

    let mut database = Database::parse(&fs::read_to_string(args.database)?)?;
    let guesses = wordsolve::parse_guesses(&args.filters, ";")?;
    database.candidates = wordsolve::filter_candidates(&guesses, &database.candidates);
    if let Some(limit) = args.truncate_candidates {
        if limit < database.candidates.len() {
            database.candidates.resize(limit, Default::default());
        }
    }
    if database.candidates.is_empty() {
        return Err(error("no solution candidates remain"));
    }
    database.candidates.sort();
    database.noncandidates.sort();

    let t0 = time::Instant::now();
    let matrix = Matrix::build(&database, &mut wordsolve::update_stderr_progress)?;
    eprintln!("preprocess time = {:?}", t0.elapsed());
    let candidates: Vec<WordId> = (0..database.candidates.len()).map(From::from).collect();
    let queries: Vec<WordId> = (0..matrix.words.len()).map(From::from).collect();

    let solver = Solver {
        dk_trunc: args.dk_trunc,
    };
    eprintln!(
        "nc = {}, nq = {}, {:?}",
        database.candidates.len(),
        database.candidates.len() + database.noncandidates.len(),
        solver,
    );
    //    let mut out_file = io::BufWriter::new(fs::File::create(&args.out_strategy)?);
    eprintln!("Solving...");
    let t0 = time::Instant::now();
    let (score, strategy) = solver.solve(
        &matrix,
        &queries,
        &candidates,
        &mut wordsolve::update_stderr_progress,
    );
    eprint!("\x1b[2K\r");
    eprintln!("\x1b[35mSOLVE TIME = {:?}\x1b[0m", t0.elapsed());
    eprintln!("===\nWORD\tD_MAX\tD_AVG");
    eprintln!(
        "{}\t{}\t{}\t{}/{}",
        match strategy {
            Some(strategy) => matrix.word(strategy.query),
            None => "?????",
        },
        score.depth_max,
        score.depth_sum as f64 / score.num_candidates as f64,
        score.depth_sum,
        score.num_candidates,
    );

    Ok(())
}
