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
    #[clap(long, short = 'o')]
    out_strategy: PathBuf,
    #[clap(long)]
    exclude_nonsolutions: bool,
    /// Filter candidates by responses. Example: -f 'hello 01001; world 02200'
    #[clap(long, short = 'f', default_value = "")]
    filters: String,
    #[clap(long, default_value = "1")]
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

    let mut database = Database::parse(&fs::read_to_string(args.database)?)?;
    let guesses = wordsolve::parse_guesses(&args.filters, ";")?;
    database.solutions = wordsolve::filter_candidates(&guesses, &database.solutions);
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
    database.nonsolutions.sort();
    println!(
        "nc = {}, nq = {}",
        database.solutions.len(),
        database.solutions.len() + database.nonsolutions.len()
    );

    let t0 = time::Instant::now();
    let matrix = Matrix::build(&database, &mut wordsolve::update_stderr_progress)?;
    eprintln!("preprocess time = {:?}", t0.elapsed());
    let candidates: Vec<WordId> = (0..database.solutions.len()).map(From::from).collect();
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
    let t0 = time::Instant::now();
    solver.dump_strategy(&matrix, &queries, root_queries, &candidates, &mut out_file)?;
    eprint!("\x1b[2K\r");
    eprintln!("solve time = {:?}", t0.elapsed());

    Ok(())
}
