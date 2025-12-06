import fs from 'fs';
import path from 'path';
import db from './db';
import { pipeline } from '@xenova/transformers';

// Configuration
const VAULT_DIR = path.join(process.cwd(), 'vault');

// Recursive file scanner
const getFiles = (dir: string): string[] => {
  const dirents = fs.readdirSync(dir, { withFileTypes: true });
  const files = dirents.map((dirent) => {
    const res = path.resolve(dir, dirent.name);
    return dirent.isDirectory() ? getFiles(res) : res;
  });
  return Array.prototype.concat(...files);
};

export const ingest = async () => {
  // 1. Hygiene: Clear the deck
  console.log('--- Purging Database ---');
  db.prepare('DELETE FROM embeddings').run();
  db.prepare('DELETE FROM atoms').run();
  
  // 2. Initialize AI
  console.log('--- Initializing Semantic Core ---');
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  });

  // 3. Scan the Vault
  console.log(`--- Scanning Sector: ${VAULT_DIR} ---`);
  if (!fs.existsSync(VAULT_DIR)) {
    console.error(`Error: Vault directory not found at ${VAULT_DIR}`);
    return;
  }

  const allFiles = getFiles(VAULT_DIR);
  const targetFiles = allFiles.filter(f => f.endsWith('.md') || f.endsWith('.txt'));
  
  console.log(`Targets Acquired: ${targetFiles.length} documents.`);

  // Prepare DB Statements
  // Note: We added 'subnode_id' to the schema earlier, but we'll leave it NULL for now.
  // We strictly store content and create the embedding.
  const insertAtom = db.prepare(`INSERT INTO atoms (content) VALUES (?)`);
  const insertEmbedding = db.prepare(`INSERT INTO embeddings (atom_id, vector) VALUES (?, ?)`);

  let totalAtoms = 0;

  // 4. Process Files
  for (const filePath of targetFiles) {
    const fileName = path.basename(filePath);
    console.log(`Processing: ${fileName}`);
    
    const content = fs.readFileSync(filePath, 'utf-8');
    
    // Split by blank lines (The Atomizer)
    const rawAtoms = content.split(/\n\s*\n+/);

    for (const text of rawAtoms) {
      if (text.trim().length < 10) continue; // Ignore tiny fragments/noise
      
      const cleanText = text.trim();

      // A. Generate Vector (Async)
      const output = await embedder(cleanText, { pooling: 'mean', normalize: true });
      const vectorBuffer = Buffer.from((output.data as Float32Array).buffer);

      // B. Write to DB (Sync)
      const info = insertAtom.run(cleanText);
      insertEmbedding.run(info.lastInsertRowid, vectorBuffer);
      
      totalAtoms++;
    }
  }

  console.log(`--- Mission Complete. ${totalAtoms} Atoms Vectorized. ---`);
};