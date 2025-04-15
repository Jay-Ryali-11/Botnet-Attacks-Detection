-- Create user 'codexslayer' with password 'code@123' only if it doesn't exist
CREATE USER IF NOT EXISTS 'codexslayer'@'localhost' IDENTIFIED BY 'code@123';

-- Create database 'botnetattack' only if it doesn't exist
CREATE DATABASE IF NOT EXISTS botnetattack;

-- Grant privileges for 'codexslayer' on 'botnetattack' only if the user exists
GRANT ALL PRIVILEGES ON botnetattack.* TO 'codexslayer'@'localhost';

-- Switch to the 'botnetattack' database
USE botnetattack;

-- Create 'users' table only if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    phonenumber VARCHAR(15) NOT NULL,
    address TEXT NOT NULL
);

-- Ensure privileges are refreshed
FLUSH PRIVILEGES;

-- Exit MySQL
EXIT;
