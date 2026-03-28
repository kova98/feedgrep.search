use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tantivy::collector::TopDocs;
use tantivy::query::{EnableScoring, QueryParser};
use tantivy::schema::{FAST, INDEXED, STORED, STRING, Schema, TEXT, TantivyDocument, Value};
use tantivy::{DocAddress, DocSet, TERMINATED};
use tantivy::{Index, IndexWriter, ReloadPolicy, doc};
use walkdir::WalkDir;
use zstd::stream::read::Decoder;

#[derive(Copy, Clone, Debug, Deserialize, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InputKind {
    Post,
    Comment,
}

#[derive(Debug, Clone)]
pub struct DiscoveredIndex {
    pub path: PathBuf,
    pub kind_hint: Option<InputKind>,
    pub year_month: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArcticShiftPost {
    id: String,
    subreddit: String,
    author: Option<String>,
    title: Option<String>,
    selftext: Option<String>,
    created_utc: i64,
}

#[derive(Debug, Deserialize)]
struct ArcticShiftComment {
    id: String,
    subreddit: String,
    author: Option<String>,
    body: Option<String>,
    created_utc: i64,
}

#[derive(Debug, Deserialize)]
struct DataEnvelope {
    data: Vec<JsonValue>,
}

#[derive(Debug, Clone)]
pub struct Fields {
    pub id: tantivy::schema::Field,
    pub kind: tantivy::schema::Field,
    pub subreddit: tantivy::schema::Field,
    pub author: tantivy::schema::Field,
    pub created_at: tantivy::schema::Field,
    pub title: tantivy::schema::Field,
    pub body: tantivy::schema::Field,
    pub combined_text: tantivy::schema::Field,
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub rank: usize,
    pub score: f32,
    pub id: String,
    pub kind: String,
    pub year_month: Option<String>,
    pub subreddit: String,
    pub author: String,
    pub created_at: i64,
    pub title: String,
    pub body: String,
}

pub fn build_schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let id = builder.add_text_field("id", STRING | STORED);
    let kind = builder.add_text_field("kind", STRING | STORED);
    let subreddit = builder.add_text_field("subreddit", STRING | STORED);
    let author = builder.add_text_field("author", STRING | STORED);
    let created_at = builder.add_i64_field("created_at", INDEXED | FAST | STORED);
    let title = builder.add_text_field("title", TEXT | STORED);
    let body = builder.add_text_field("body", TEXT | STORED);
    let combined_text = builder.add_text_field("combined_text", TEXT);
    let schema = builder.build();

    (
        schema,
        Fields {
            id,
            kind,
            subreddit,
            author,
            created_at,
            title,
            body,
            combined_text,
        },
    )
}

pub fn fields_from_schema(schema: &Schema) -> Result<Fields> {
    Ok(Fields {
        id: schema.get_field("id").context("missing field id")?,
        kind: schema.get_field("kind").context("missing field kind")?,
        subreddit: schema.get_field("subreddit").context("missing field subreddit")?,
        author: schema.get_field("author").context("missing field author")?,
        created_at: schema.get_field("created_at").context("missing field created_at")?,
        title: schema.get_field("title").context("missing field title")?,
        body: schema.get_field("body").context("missing field body")?,
        combined_text: schema
            .get_field("combined_text")
            .context("missing field combined_text")?,
    })
}

pub fn open_or_create_index(index_dir: &Path, schema: Schema) -> Result<Index> {
    std::fs::create_dir_all(index_dir).with_context(|| format!("create {}", index_dir.display()))?;
    match Index::open_in_dir(index_dir) {
        Ok(index) => Ok(index),
        Err(_) => Index::create_in_dir(index_dir, schema).context("create tantivy index"),
    }
}

pub fn discover_input_files(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in inputs {
        if input.is_file() {
            files.push(input.clone());
            continue;
        }
        if input.is_dir() {
            for entry in WalkDir::new(input).into_iter().filter_map(Result::ok) {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("zst") {
                    files.push(path.to_path_buf());
                }
            }
            continue;
        }
        bail!("input path does not exist: {}", input.display());
    }
    files.sort();
    Ok(files)
}

pub fn discover_indexes(index_root: &Path) -> Result<Vec<DiscoveredIndex>> {
    if !index_root.is_dir() {
        bail!("index root not found: {}", index_root.display());
    }

    let mut indexes = Vec::new();
    for entry in std::fs::read_dir(index_root)
        .with_context(|| format!("read index root {}", index_root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if Index::open_in_dir(&path).is_err() {
            continue;
        }

        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        indexes.push(DiscoveredIndex {
            path,
            kind_hint: infer_kind_from_index_name(&name),
            year_month: infer_year_month_from_index_name(&name),
        });
    }

    indexes.sort_by(|left, right| left.path.cmp(&right.path));
    if indexes.is_empty() {
        bail!("no tantivy indexes found under {}", index_root.display());
    }
    Ok(indexes)
}

pub fn resolve_reddit_archive(root: &Path, kind: InputKind, year_month: &str) -> Result<PathBuf> {
    if !valid_year_month(year_month) {
        bail!("invalid --year-month format: {year_month}; expected YYYY-MM");
    }

    let subdir = match kind {
        InputKind::Post => "submissions",
        InputKind::Comment => "comments",
    };
    let prefix = match kind {
        InputKind::Post => "RS",
        InputKind::Comment => "RC",
    };

    let path = root.join(subdir).join(format!("{prefix}_{year_month}.zst"));
    if !path.is_file() {
        bail!("archive file not found: {}", path.display());
    }
    Ok(path)
}

pub fn valid_year_month(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 7
        && bytes[0..4].iter().all(|b| b.is_ascii_digit())
        && bytes[4] == b'-'
        && bytes[5..7].iter().all(|b| b.is_ascii_digit())
}

pub fn resolve_kind(cli_kind: Option<InputKind>, path: &Path) -> Result<InputKind> {
    if let Some(kind) = cli_kind {
        return Ok(kind);
    }

    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
    if file_name.starts_with("RC_") {
        return Ok(InputKind::Comment);
    }
    if file_name.starts_with("RS_") {
        return Ok(InputKind::Post);
    }

    bail!(
        "could not infer kind from file name {}; use --kind post or --kind comment",
        path.display()
    )
}

pub fn resolve_index_dir(
    index_dir: Option<PathBuf>,
    index_root: Option<PathBuf>,
    input: &Path,
    kind: InputKind,
) -> Result<PathBuf> {
    if let Some(index_dir) = index_dir {
        return Ok(index_dir);
    }

    let index_root = index_root.unwrap_or_else(|| PathBuf::from("./tantivy-indexes"));
    let base_name = input
        .file_name()
        .and_then(|s| s.to_str())
        .map(strip_all_extensions)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "archive".to_string());

    Ok(index_root.join(format!("{}-{}", kind.as_str(), base_name)))
}

pub fn index_file(
    path: &Path,
    kind: InputKind,
    fields: &Fields,
    writer: &mut IndexWriter,
    commit_every: usize,
) -> Result<usize> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let decoder = Decoder::new(file).with_context(|| format!("open zstd decoder {}", path.display()))?;
    let reader = BufReader::new(decoder);

    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value: JsonValue =
            serde_json::from_str(trimmed).with_context(|| format!("parse json line in {}", path.display()))?;

        count += index_value(value, kind, fields, writer)?;
        if count > 0 && count % commit_every == 0 {
            writer.commit().context("intermediate commit")?;
            println!("file={} indexed_docs={count}", path.display());
        }
    }

    Ok(count)
}

fn index_value(value: JsonValue, kind: InputKind, fields: &Fields, writer: &mut IndexWriter) -> Result<usize> {
    match value {
        JsonValue::Object(obj) if obj.contains_key("data") => {
            let envelope: DataEnvelope =
                serde_json::from_value(JsonValue::Object(obj)).context("decode data envelope")?;
            let mut count = 0usize;
            for item in envelope.data {
                count += index_record(item, kind, fields, writer)?;
            }
            Ok(count)
        }
        JsonValue::Array(items) => {
            let mut count = 0usize;
            for item in items {
                count += index_record(item, kind, fields, writer)?;
            }
            Ok(count)
        }
        other => index_record(other, kind, fields, writer),
    }
}

fn index_record(value: JsonValue, kind: InputKind, fields: &Fields, writer: &mut IndexWriter) -> Result<usize> {
    let document = match kind {
        InputKind::Post => {
            let post: ArcticShiftPost = serde_json::from_value(value).context("decode post")?;
            build_post_doc(post, fields)
        }
        InputKind::Comment => {
            let comment: ArcticShiftComment =
                serde_json::from_value(value).context("decode comment")?;
            build_comment_doc(comment, fields)
        }
    };

    writer.add_document(document)?;
    Ok(1)
}

fn build_post_doc(post: ArcticShiftPost, fields: &Fields) -> TantivyDocument {
    let title = post.title.unwrap_or_default();
    let body = post.selftext.unwrap_or_default();
    let combined = normalize_text(&title, &body);

    doc!(
        fields.id => post.id,
        fields.kind => "post",
        fields.subreddit => post.subreddit,
        fields.author => post.author.unwrap_or_default(),
        fields.created_at => post.created_utc,
        fields.title => title,
        fields.body => body,
        fields.combined_text => combined,
    )
}

fn build_comment_doc(comment: ArcticShiftComment, fields: &Fields) -> TantivyDocument {
    let body = comment.body.unwrap_or_default();
    let combined = normalize_text("", &body);

    doc!(
        fields.id => comment.id,
        fields.kind => "comment",
        fields.subreddit => comment.subreddit,
        fields.author => comment.author.unwrap_or_default(),
        fields.created_at => comment.created_utc,
        fields.title => "",
        fields.body => body,
        fields.combined_text => combined,
    )
}

fn normalize_text(title: &str, body: &str) -> String {
    let mut out = String::new();
    if !title.trim().is_empty() {
        out.push_str(title.trim());
    }
    if !body.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(body.trim());
    }
    out
}

pub fn search_index(
    index_dir: &Path,
    query_text: &str,
    limit: usize,
    year_month: Option<&str>,
) -> Result<Vec<SearchHit>> {
    let index = Index::open_in_dir(index_dir)
        .with_context(|| format!("open tantivy index {}", index_dir.display()))?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .context("create tantivy reader")?;
    let searcher = reader.searcher();
    let fields = fields_from_schema(&index.schema())?;

    let query_parser = QueryParser::for_index(
        &index,
        vec![fields.title, fields.body, fields.combined_text, fields.subreddit],
    );
    let query = query_parser
        .parse_query(query_text)
        .with_context(|| format!("parse query {query_text}"))?;

    let top_docs = searcher
        .search(&query, &TopDocs::with_limit(limit))
        .context("search tantivy index")?;

    let mut hits = Vec::with_capacity(top_docs.len());
    for (rank, (score, addr)) in top_docs.into_iter().enumerate() {
        let doc: TantivyDocument = searcher.doc(addr).context("load document")?;
        hits.push(SearchHit {
            rank: rank + 1,
            score,
            id: doc_first_text(&doc, fields.id),
            kind: doc_first_text(&doc, fields.kind),
            year_month: year_month.map(ToOwned::to_owned),
            subreddit: doc_first_text(&doc, fields.subreddit),
            author: doc_first_text(&doc, fields.author),
            created_at: doc_first_i64(&doc, fields.created_at),
            title: doc_first_text(&doc, fields.title),
            body: doc_first_text(&doc, fields.body),
        });
    }

    Ok(hits)
}

pub fn stream_index_hits<F>(
    index_dir: &Path,
    query_text: &str,
    year_month: Option<&str>,
    mut on_hit: F,
) -> Result<usize>
where
    F: FnMut(SearchHit) -> Result<()>,
{
    let index = Index::open_in_dir(index_dir)
        .with_context(|| format!("open tantivy index {}", index_dir.display()))?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .context("create tantivy reader")?;
    let searcher = reader.searcher();
    let fields = fields_from_schema(&index.schema())?;

    let query_parser = QueryParser::for_index(
        &index,
        vec![fields.title, fields.body, fields.combined_text, fields.subreddit],
    );
    let query = query_parser
        .parse_query(query_text)
        .with_context(|| format!("parse query {query_text}"))?;
    let weight = query
        .weight(EnableScoring::enabled_from_searcher(&searcher))
        .context("build streaming query weight")?;

    let mut emitted = 0usize;
    for (segment_ord, segment_reader) in searcher.segment_readers().iter().enumerate() {
        let mut scorer = weight.scorer(segment_reader, 1.0)?;
        let mut doc = scorer.doc();
        while doc != TERMINATED {
            emitted += 1;
            let doc_address = DocAddress::new(segment_ord as u32, doc);
            let hit_doc: TantivyDocument = searcher.doc(doc_address).context("load streamed document")?;
            on_hit(SearchHit {
                rank: emitted,
                score: scorer.score(),
                id: doc_first_text(&hit_doc, fields.id),
                kind: doc_first_text(&hit_doc, fields.kind),
                year_month: year_month.map(ToOwned::to_owned),
                subreddit: doc_first_text(&hit_doc, fields.subreddit),
                author: doc_first_text(&hit_doc, fields.author),
                created_at: doc_first_i64(&hit_doc, fields.created_at),
                title: doc_first_text(&hit_doc, fields.title),
                body: doc_first_text(&hit_doc, fields.body),
            })?;
            doc = scorer.advance();
        }
    }

    Ok(emitted)
}

pub fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}...")
}

fn doc_first_text(doc: &TantivyDocument, field: tantivy::schema::Field) -> String {
    doc.get_first(field)
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string()
}

fn doc_first_i64(doc: &TantivyDocument, field: tantivy::schema::Field) -> i64 {
    doc.get_first(field)
        .and_then(|value| value.as_i64())
        .unwrap_or_default()
}

fn strip_all_extensions(file_name: &str) -> String {
    let mut out = file_name.to_string();
    while let Some((stem, _ext)) = out.rsplit_once('.') {
        out = stem.to_string();
    }
    out
}

fn infer_kind_from_index_name(name: &str) -> Option<InputKind> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("comment") || name.contains("RC_") {
        return Some(InputKind::Comment);
    }
    if lower.contains("submission") || lower.contains("post") || name.contains("RS_") {
        return Some(InputKind::Post);
    }
    None
}

fn infer_year_month_from_index_name(name: &str) -> Option<String> {
    let chars: Vec<char> = name.chars().collect();
    if chars.len() < 7 {
        return None;
    }

    for start in 0..=chars.len() - 7 {
        let slice = &chars[start..start + 7];
        if slice[0..4].iter().all(|c| c.is_ascii_digit())
            && slice[4] == '-'
            && slice[5..7].iter().all(|c| c.is_ascii_digit())
        {
            return Some(slice.iter().collect());
        }
    }
    None
}

impl InputKind {
    pub fn as_str(self) -> &'static str {
        match self {
            InputKind::Post => "post",
            InputKind::Comment => "comment",
        }
    }
}
