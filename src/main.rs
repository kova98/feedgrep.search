use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Args, Parser, Subcommand};
use feedgrep_search::{
    InputKind, build_schema, discover_input_files, index_file, open_or_create_index,
    resolve_index_dir, resolve_kind, resolve_reddit_archive, search_index, stream_index_hits,
    truncate,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

const DEFAULT_LIMIT: usize = 10;
const MAX_LIMIT: usize = 1_000;
const DEFAULT_MIN_SCORE: f32 = 10.0;

#[derive(Parser, Debug)]
#[command(name = "feedgrep-search")]
#[command(about = "Build, query, and serve Tantivy indexes for feedgrep")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Index(IndexArgs),
    Search(SearchArgs),
    Serve(ServeArgs),
}

#[derive(Args, Debug)]
struct IndexArgs {
    #[arg(long, conflicts_with = "index_root")]
    index_dir: Option<PathBuf>,

    #[arg(long, conflicts_with = "index_dir")]
    index_root: Option<PathBuf>,

    #[arg(long, value_enum)]
    kind: Option<InputKind>,

    #[arg(long)]
    reddit_root: Option<PathBuf>,

    #[arg(long)]
    year_month: Option<String>,

    #[arg(long, default_value_t = 50_000_000)]
    writer_memory_bytes: usize,

    #[arg(long, default_value_t = 5_000)]
    commit_every: usize,

    #[arg()]
    inputs: Vec<PathBuf>,
}

#[derive(Args, Debug)]
struct SearchArgs {
    #[arg(long, conflicts_with = "index_root")]
    index_dir: Option<PathBuf>,

    #[arg(long, conflicts_with = "index_dir")]
    index_root: Option<PathBuf>,

    #[arg(long, requires = "year_month")]
    kind: Option<InputKind>,

    #[arg(long, requires = "kind")]
    year_month: Option<String>,

    #[arg(long, default_value_t = 10)]
    limit: usize,

    query: String,
}

#[derive(Args, Debug)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:4001")]
    listen_addr: SocketAddr,

    #[arg(long, default_value = "./indexes.json")]
    indexes_config: PathBuf,
}

#[derive(Clone)]
struct AppState {
    indexes: IndexManifest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchRequest {
    kind: Option<InputKind>,
    query: String,
    limit: Option<usize>,
    min_score: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResponse {
    searched_indexes: usize,
    hit_count: usize,
    hits: Vec<feedgrep_search::SearchHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StreamEnd {
    searched_indexes: usize,
    hit_count: usize,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    ok: bool,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Clone, Debug, Deserialize)]
struct IndexManifest {
    indexes: Vec<IndexEntry>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IndexEntry {
    kind: InputKind,
    year_month: String,
    path: PathBuf,
}

struct ApiError(anyhow::Error);

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index(args) => run_index(args),
        Command::Search(args) => run_search(args),
        Command::Serve(args) => run_serve(args).await,
    }
}

fn run_index(args: IndexArgs) -> Result<()> {
    let files = resolve_inputs(&args)?;
    if files.is_empty() {
        bail!("no input files found");
    }

    if args.index_dir.is_some() && files.len() != 1 {
        bail!("--index-dir can only be used with a single input file; use --index-root for multiple files");
    }

    let (schema, fields) = build_schema();
    let mut total_indexed_docs = 0usize;

    for file in files {
        let kind = resolve_kind(args.kind, &file)?;
        let index_dir = resolve_index_dir(args.index_dir.clone(), args.index_root.clone(), &file, kind)?;
        let index = open_or_create_index(&index_dir, schema.clone())?;
        let mut writer = index
            .writer(args.writer_memory_bytes)
            .context("create tantivy writer")?;

        let indexed_docs = index_file(&file, kind, &fields, &mut writer, args.commit_every)
            .with_context(|| format!("index file {}", file.display()))?;
        writer.commit().context("final commit")?;

        total_indexed_docs += indexed_docs;
        println!(
            "completed file={} kind={} index_dir={} indexed_docs={}",
            file.display(),
            kind.as_str(),
            index_dir.display(),
            indexed_docs
        );
    }

    println!("total_indexed_docs={total_indexed_docs}");
    Ok(())
}

fn run_search(args: SearchArgs) -> Result<()> {
    let index_dir = match args.index_dir {
        Some(path) => path,
        None => resolve_cli_search_index_dir(args.index_root, args.kind, args.year_month.as_deref())?,
    };
    let hits = search_index(&index_dir, &args.query, args.limit, args.year_month.as_deref())?;

    for hit in hits {
        println!("rank={}", hit.rank);
        println!("score={}", hit.score);
        println!("id={}", hit.id);
        println!("kind={}", hit.kind);
        println!("subreddit={}", hit.subreddit);
        println!("author={}", hit.author);
        println!("created_at={}", hit.created_at);
        println!("title={}", hit.title);
        println!("body={}", truncate(&hit.body, 400));
        println!("---");
    }

    Ok(())
}

async fn run_serve(args: ServeArgs) -> Result<()> {
    let indexes = load_indexes_config(&args.indexes_config)?;
    let app_state = Arc::new(AppState {
        indexes,
    });

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/search", post(search))
        .route("/stream", post(stream))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(args.listen_addr)
        .await
        .with_context(|| format!("bind {}", args.listen_addr))?;
    println!("listening addr={}", args.listen_addr);
    axum::serve(listener, app).await.context("serve axum")?;
    Ok(())
}

async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse { ok: true })
}

async fn search(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let limit = sanitize_limit(request.limit)?;
    let min_score = sanitize_min_score(request.min_score)?;
    let mut hits = Vec::new();
    let mut searched_indexes = 0usize;

    for entry in matching_indexes(&state, request.kind)? {
        searched_indexes += 1;
        hits.extend(
            search_index(&entry.path, &request.query, limit, Some(&entry.year_month))?
                .into_iter()
                .filter(|hit| hit.score >= min_score),
        );
    }

    if searched_indexes == 0 {
        return Err(ApiError(anyhow::anyhow!("no configured indexes matched the request")));
    }

    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.truncate(limit);
    for (idx, hit) in hits.iter_mut().enumerate() {
        hit.rank = idx + 1;
    }

    Ok(Json(SearchResponse {
        searched_indexes,
        hit_count: hits.len(),
        hits,
    }))
}

async fn stream(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SearchRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let indexes = matching_indexes(&state, request.kind)?;
    let min_score = sanitize_min_score(request.min_score)?;
    let query = request.query;
    let (sender, receiver) = mpsc::channel::<StreamEnvelope>(32);

    tokio::task::spawn_blocking(move || {
        let mut searched_indexes = 0usize;
        let mut hit_count = 0usize;

        for entry in indexes {
            searched_indexes += 1;
            let send_result = stream_index_hits(&entry.path, &query, Some(&entry.year_month), |mut hit| {
                if hit.score >= min_score {
                    hit_count += 1;
                    hit.rank = hit_count;
                    sender
                        .blocking_send(StreamEnvelope::Hit(hit))
                        .map_err(|_| anyhow::anyhow!("stream receiver dropped"))?;
                }
                Ok(())
            });

            if let Err(err) = send_result {
                let _ = sender.blocking_send(StreamEnvelope::Error(err.to_string()));
                return;
            }
        }

        let _ = sender.blocking_send(StreamEnvelope::End(StreamEnd {
            searched_indexes,
            hit_count,
        }));
    });

    let stream = ReceiverStream::new(receiver).map(|message| {
        let event = match message {
            StreamEnvelope::Hit(hit) => Event::default()
                .event("hit")
                .json_data(hit)
                .expect("serialize hit event"),
            StreamEnvelope::End(summary) => Event::default()
                .event("end")
                .json_data(summary)
                .expect("serialize end event"),
            StreamEnvelope::Error(error) => Event::default().event("error").data(error),
        };
        Ok(event)
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

fn sanitize_limit(limit: Option<usize>) -> Result<usize, ApiError> {
    let limit = limit.unwrap_or(DEFAULT_LIMIT);
    if limit == 0 {
        return Err(ApiError(anyhow::anyhow!("limit must be greater than 0")));
    }
    if limit > MAX_LIMIT {
        return Err(ApiError(anyhow::anyhow!(
            "limit {} exceeds max supported limit {}",
            limit,
            MAX_LIMIT
        )));
    }
    Ok(limit)
}

fn sanitize_min_score(min_score: Option<f32>) -> Result<f32, ApiError> {
    let min_score = min_score.unwrap_or(DEFAULT_MIN_SCORE);
    if !min_score.is_finite() {
        return Err(ApiError(anyhow::anyhow!("minScore must be a finite number")));
    }
    Ok(min_score)
}

fn resolve_inputs(args: &IndexArgs) -> Result<Vec<PathBuf>> {
    if let (Some(root), Some(kind), Some(year_month)) = (&args.reddit_root, args.kind, &args.year_month) {
        if !args.inputs.is_empty() {
            bail!("do not pass positional inputs when using --reddit-root with --kind and --year-month");
        }
        let path = resolve_reddit_archive(root, kind, year_month)?;
        return Ok(vec![path]);
    }

    if args.reddit_root.is_some() || args.year_month.is_some() {
        bail!("--reddit-root and --year-month must be used together with --kind");
    }

    discover_input_files(&args.inputs)
}

fn load_indexes_config(path: &PathBuf) -> Result<IndexManifest> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read indexes config {}", path.display()))?;
    let manifest: IndexManifest =
        serde_json::from_str(&raw).with_context(|| format!("parse indexes config {}", path.display()))?;
    if manifest.indexes.is_empty() {
        bail!("indexes config {} contains no indexes", path.display());
    }
    Ok(manifest)
}

fn matching_indexes(state: &AppState, kind: Option<InputKind>) -> Result<Vec<IndexEntry>, ApiError> {
    let indexes: Vec<IndexEntry> = state
        .indexes
        .indexes
        .iter()
        .filter(|entry| kind.is_none_or(|kind| entry.kind.as_str() == kind.as_str()))
        .cloned()
        .collect();

    if indexes.is_empty() {
        return Err(ApiError(anyhow::anyhow!("no configured indexes matched the request")));
    }

    Ok(indexes)
}

fn resolve_cli_search_index_dir(
    index_root: Option<PathBuf>,
    kind: Option<InputKind>,
    year_month: Option<&str>,
) -> Result<PathBuf> {
    let (kind, year_month) = match (kind, year_month) {
        (Some(kind), Some(year_month)) => (kind, year_month),
        _ => bail!("search requires either --index-dir or both --index-root, --kind, and --year-month"),
    };
    let index_root = index_root.unwrap_or_else(|| PathBuf::from("./tantivy-indexes"));
    let suffix = match kind {
        InputKind::Post => format!("RS_{year_month}"),
        InputKind::Comment => format!("RC_{year_month}"),
    };
    Ok(index_root.join(format!("{}-{}", kind.as_str(), suffix)))
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            error: self.0.to_string(),
        });
        (StatusCode::BAD_REQUEST, body).into_response()
    }
}

impl<E> From<E> for ApiError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self(error.into())
    }
}

enum StreamEnvelope {
    Hit(feedgrep_search::SearchHit),
    End(StreamEnd),
    Error(String),
}
