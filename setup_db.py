import os
import sys
import getpass
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def require_env(key):
    val = os.environ.get(key, '').strip()
    if not val:
        print(f"\n  [ERROR] '{key}' is missing or empty in your .env file.")
        print(  f"  Add it and try again.\n")
        sys.exit(1)
    return val


def run():
    print("\n=== Botnet Attack Detection — Local Database Setup ===\n")
    print("  Reading credentials from .env ...\n")

    db_host     = require_env('DB_HOST')
    db_port     = int(require_env('DB_PORT'))
    db_name     = require_env('DB_NAME')
    db_user     = require_env('DB_USER')
    db_password = require_env('DB_PASSWORD')

    root_password = os.environ.get('MYSQL_ROOT_PASSWORD', '').strip()
    if not root_password:
        print("  MYSQL_ROOT_PASSWORD not in .env.")
        root_password = getpass.getpass("  Enter MySQL root password: ")

    steps = [
        (
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
            f"Database '{db_name}' ensured."
        ),
        (
            f"CREATE USER IF NOT EXISTS '{db_user}'@'%' "
            f"IDENTIFIED BY '{db_password}'",
            f"User '{db_user}' ensured."
        ),
        (
            f"GRANT SELECT, INSERT, UPDATE, DELETE "
            f"ON `{db_name}`.* TO '{db_user}'@'%'",
            f"Least-privilege grants applied."
        ),
        (
            "FLUSH PRIVILEGES",
            "Privileges flushed."
        ),
        (
            f"USE `{db_name}`",
            f"Switched to database '{db_name}'."
        ),
        (
            """CREATE TABLE IF NOT EXISTS users (
                id          INT          NOT NULL AUTO_INCREMENT,
                name        VARCHAR(100) NOT NULL,
                email       VARCHAR(100) NOT NULL,
                password    VARCHAR(255) NOT NULL,
                phonenumber VARCHAR(15)  NOT NULL,
                address     TEXT         NOT NULL,
                created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                UNIQUE KEY uq_email       (email),
                UNIQUE KEY uq_phonenumber (phonenumber)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
            "Table 'users' ensured."
        ),
    ]

    try:
        conn = mysql.connector.connect(
            host=db_host,
            port=db_port,
            user='root',
            passwd=root_password,
            ssl_disabled=True
        )
        cursor = conn.cursor()

        print()
        for sql, message in steps:
            cursor.execute(sql)
            conn.commit()
            print(f"  OK  {message}")

        cursor.close()
        conn.close()

        print("\n  Setup complete.")
        print("  Run the app with: python run.py\n")

    except mysql.connector.Error as e:
        print(f"\n  [ERROR] {e}")
        print("  Check your .env values and MySQL root password.\n")
        sys.exit(1)


if __name__ == "__main__":
    run()