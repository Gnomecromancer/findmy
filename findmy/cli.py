"""find CLI — index your machine, search it."""
from __future__ import annotations

import sys
from pathlib import Path

import click


@click.group()
def main():
    """find — semantic search across your local files."""


@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--exclude", "-e", multiple=True, help="Extra dir names to exclude.")
@click.option("--force", "-f", is_flag=True, help="Re-index even unchanged files.")
@click.option("--quiet", "-q", is_flag=True, help="No progress output.")
def index(paths, exclude, force, quiet):
    """Index files under PATH(s). Defaults to home directory."""
    from .indexer import index as do_index

    roots = list(paths) or [Path.home()]

    for root in roots:
        if not quiet:
            click.echo(f"Indexing {root} …")

        last_reported = [0]

        def progress(path_str: str, current: int, total: int):
            # Report every 50 files or on last file
            if not quiet and (current % 50 == 0 or current == total):
                pct = int(100 * current / max(total, 1))
                click.echo(f"  [{pct:3d}%] {current}/{total}  {Path(path_str).name}", nl=True)
            last_reported[0] = current

        try:
            result = do_index(root, list(exclude), force=force, on_progress=progress if not quiet else None)
        except KeyboardInterrupt:
            if not quiet:
                click.echo("\nInterrupted — progress saved. Resume by running index again.")
            return

        if not quiet:
            click.echo(
                f"Done — {result['processed']} files, "
                f"{result['chunks']} chunks indexed "
                f"({result['skipped']} unchanged, skipped)."
            )


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--top", "-n", default=10, show_default=True, help="Number of results.")
@click.option("--type", "ext_filter", default="", help="Comma-separated extensions, e.g. py,md")
@click.option("--show-text", is_flag=True, help="Print matching chunk text.")
def search(query, top, ext_filter, show_text):
    """Search indexed files with a natural-language QUERY."""
    from .embedder import Embedder
    from .store import Store

    q = " ".join(query)
    exts = {f".{e.lstrip('.')}" for e in ext_filter.split(",") if e} if ext_filter else None

    embedder = Embedder()
    q_vec = embedder.embed_query(q)

    with Store() as store:
        results = store.search(q_vec, top_k=top, ext_filter=exts)

    if not results:
        click.echo("No results found. Have you run `find index` yet?")
        sys.exit(1)

    for i, r in enumerate(results, 1):
        score_pct = int(r["score"] * 100)
        click.echo(f"\n{i:2d}. [{score_pct}%] {r['path']}")
        click.echo(f"    {r['label']}")
        if show_text:
            snippet = r["text"][:300].replace("\n", " ")
            click.echo(f"    {snippet}{'…' if len(r['text']) > 300 else ''}")


@main.command()
def status():
    """Show index statistics."""
    from .store import Store
    from .config import INDEX_PATH, META_PATH

    if not INDEX_PATH.exists():
        click.echo("No index found. Run `find index` first.")
        sys.exit(1)

    with Store() as store:
        s = store.stats()

    click.echo(f"Files indexed : {s['files']:,}")
    click.echo(f"Chunks        : {s['chunks']:,}")
    click.echo(f"Vectors       : {s['vectors']:,}")
    click.echo(f"Index size    : {s['index_size_mb']:.1f} MB")
    click.echo(f"Index path    : {INDEX_PATH}")
    click.echo(f"Metadata path : {META_PATH}")


@main.command()
@click.argument("path", type=click.Path(path_type=Path))
def forget(path):
    """Remove a specific file from the index."""
    from .store import Store

    with Store() as store:
        store.delete_file(path.resolve())
        click.echo(f"Removed {path} from index.")
