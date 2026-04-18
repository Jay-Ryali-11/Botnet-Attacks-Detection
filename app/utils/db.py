import mysql.connector.pooling
from flask import current_app, g
import logging

logger = logging.getLogger(__name__)

_pool: mysql.connector.pooling.MySQLConnectionPool | None = None

def init_pool(app):
    global _pool
    _pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="botnet_pool",
        pool_size=app.config.get('DB_POOL_SIZE', 5),
        pool_reset_session=True,
        host=app.config['DB_HOST'],
        port=app.config['DB_PORT'],
        user=app.config['DB_USER'],
        passwd=app.config['DB_PASSWORD'],
        database=app.config['DB_NAME'],
        charset='utf8mb4',
        connection_timeout=30,
    )
    logger.info(
        "MySQL connection pool created — size=%d database=%s",
        app.config.get('DB_POOL_SIZE', 5),
        app.config['DB_NAME'],
    )


def get_db():
    if _pool is None:
        raise RuntimeError(
            "Database pool is not initialised. "
            "Ensure init_pool(app) is called inside create_app()."
        )
    if 'db' not in g:
        g.db = _pool.get_connection()
    return g.db


def teardown_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def execute_query(sql, params=None, fetch='one', commit=False):
    db     = get_db()
    cursor = db.cursor()
    try:
        cursor.execute(sql, params)
        if commit:
            db.commit()
            return None
        if fetch == 'one':
            return cursor.fetchone()
        if fetch == 'all':
            return cursor.fetchall()
        return None
    except Exception:
        if commit:
            db.rollback()
        logger.exception("Query failed. SQL: %.120s", sql)
        g.db = _pool.get_connection() 
        raise
    finally:
        cursor.close()