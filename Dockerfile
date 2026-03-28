FROM rust:1.87-alpine AS builder
WORKDIR /app
RUN apk add --no-cache musl-dev
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/target/release/feedgrep-search /feedgrep-search
EXPOSE 8080
CMD ["/feedgrep-search", "serve", "--listen-addr", "0.0.0.0:8080", "--index-root", "/data/reddit-indexes"]
