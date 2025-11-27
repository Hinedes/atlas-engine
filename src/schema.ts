import db from './db';

const schema = `
  /* 1. Hierarchy */
  CREATE TABLE IF NOT EXISTS spaces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE);
  CREATE TABLE IF NOT EXISTS nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, space_id INTEGER NOT NULL, name TEXT NOT NULL, FOREIGN KEY (space_id) REFERENCES spaces(id));
  CREATE TABLE IF NOT EXISTS subnodes (id INTEGER PRIMARY KEY AUTOINCREMENT, node_id INTEGER NOT NULL, name TEXT NOT NULL, FOREIGN KEY (node_id) REFERENCES nodes(id));

  /* 2. Content */
  CREATE TABLE IF NOT EXISTS atoms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );

  /* 3. The Semantic Core (Vectors) */
  /* We store the raw binary vector here. */
  CREATE TABLE IF NOT EXISTS embeddings (
    atom_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    FOREIGN KEY (atom_id) REFERENCES atoms(id) ON DELETE CASCADE
  );
`;

export const initSchema = () => {
  db.exec(schema);
  console.log('Schema loaded. Semantic Vector Store established.');
};