import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

// 1. Define the secure location for our data
const DATA_DIR = path.join(process.cwd(), '.atlas');
const DB_PATH = path.join(DATA_DIR, 'atlas.db');

// 2. Ensure the fortress directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR);
  console.log(`Created data directory at: ${DATA_DIR}`);
}

// 3. Open the connection
const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL'); // rigorous performance mode

console.log(`Database connected: ${DB_PATH}`);

export default db;