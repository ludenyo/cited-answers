"""Postgres connection helper."""

import os

import psycopg
from pgvector.psycopg import register_vector


def connect(url: str | None = None, register: bool = True) -> psycopg.Connection:
    """Open a connection using DATABASE_URL unless a URL is given.

    register=False is for the migration runner: the vector type does not exist
    until the first migration has created the extension.
    """
    conn = psycopg.connect(url or os.environ["DATABASE_URL"])
    if register:
        register_vector(conn)
    return conn
