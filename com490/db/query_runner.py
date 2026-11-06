from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import closing
from string import Template
from pathlib import Path
import pandas as pd

SQL_DIR = Path(__file__).parent.parent / "sql"

def _load_sql(relative_path: str, **kwargs) -> str:
    """Load a .sql file and substitute $placeholders with kwargs."""
    sql = (SQL_DIR / relative_path).read_text()
    return Template(sql).substitute(**kwargs)

def execute(conn, relative_path: str, **kwargs) -> None:
    """Load and execute a SQL file with no return value (DDL statements)."""
    sql = _load_sql(relative_path, **kwargs)
    with closing(conn.cursor()) as cur:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                cur.execute(statement)

def fetch(conn, sql_path: Path, **kwargs) -> pd.DataFrame:
    """Execute a SQL file and return results as a DataFrame."""
    sql = _load_sql(sql_path, **kwargs)
    with closing(conn.cursor()) as cur:
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)

def _fetch_sync(conn, sql:     str) -> pd.DataFrame:
    """Internal blocking fetch."""
    with closing(conn.cursor()) as cur:
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


_executor = ThreadPoolExecutor()


def fetch_async(conn, sql_path: Path, **kwargs) -> Future:
    """Execute a SQL file and return a Future[DataFrame].
    
    Usage:
        future = fetch(conn, SOME_QUERY, userns=userns)
        # ... submit other futures ...
        df = future.result()  # blocks only when you need the result
    """
    sql = _load_sql(sql_path, **kwargs)
    return _executor.submit(_fetch_sync, conn, sql)


def fetch_all(conn, queries: list[tuple[Path, dict]]) -> list[pd.DataFrame]:
    """Fetch multiple queries in parallel, return DataFrames in the same order.
    
    Usage:
        results = fetch_all(conn, [
            (STOPS_QUERY,  {"userns": userns}),
            (TRIPS_QUERY,  {"userns": userns}),
        ])
        stops_df, trips_df = results
    """
    futures = [fetch(conn, sql_path, **kwargs) for sql_path, kwargs in queries]
    return [f.result() for f in futures]
