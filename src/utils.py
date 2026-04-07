"""
Shared utility helpers for the Divvy bike rebalancing pipeline.

Functions
---------
connect_duckdb(file_path)
    Open a DuckDB connection and register the parquet file as a view.
load_parquet_view(con, file_path, view_name)
    Register an arbitrary parquet file as a named view in an existing connection.
get_data_path(default)
    Prompt the user for the path to the raw parquet file at runtime.
"""

import os
import duckdb


def get_data_path(default: str = "../data/raw/divvy.parquet") -> str:
    """
    Prompt the user for the path to the raw parquet file.
    Press Enter to accept the default.

    Parameters
    ----------
    default : str
        Path to use if the user presses Enter without typing anything.

    Returns
    -------
    str
        Validated path to the parquet file.

    Raises
    ------
    FileNotFoundError
        If no file exists at the resolved path.
    """
    path = input(f"Enter path to divvy.parquet [{default}]: ").strip()
    if not path:
        path = default
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at: {path}")
    return path


def connect_duckdb(file_path: str):
    """
    Open a DuckDB in-memory connection and register the parquet file as a view
    named 'trips'.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the divvy.parquet file.

    Returns
    -------
    duckdb.DuckDBPyConnection
        An open DuckDB connection with the 'trips' view registered.
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW trips AS SELECT * FROM read_parquet('{file_path}')")
    return con


def load_parquet_view(con, file_path: str, view_name: str) -> None:
    """
    Register an arbitrary parquet file as a named view in an existing DuckDB
    connection.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        An open DuckDB connection.
    file_path : str
        Path to the parquet file to register.
    view_name : str
        Name to assign to the view inside DuckDB.
    """
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{file_path}')")
