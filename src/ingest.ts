import fs from 'fs';
import path from 'path';
import db from './db';
import { embedTextAsBuffer, getEmbedder } from './embedder';

const VAULT_DIR = path.join(process.cwd(), 'vault');

interface AtomResult {
  id: number | bigint;
  content: string;
}

function loadDocument(filename: string): string[] {
  const filePath = path.join(VAULT_DIR, filename);
  
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  return content
    .split(/\n\s*\n+/)
    .map(text => text.trim())
    .filter(text => text.length > 0);
}

// Prepare statements once at module level for reuse
const insertAtomStmt = db.prepare(`INSERT INTO atoms (content) VALUES (?)`);
const insertEmbeddingStmt = db.prepare(`INSERT INTO embeddings (atom_id, vector) VALUES (?, ?)`);

function saveAtom(content: string, vector: Buffer): AtomResult {
  const info = insertAtomStmt.run(content);
  insertEmbeddingStmt.run(info.lastInsertRowid, vector);

  return { id: info.lastInsertRowid, content };
}

export async function ingest(filename = 'briefing.md'): Promise<void> {
  try {
    // Initialize embedder (uses singleton pattern, so safe to call multiple times)
    await getEmbedder();
    const atoms = loadDocument(filename);

    console.log(`Processing ${atoms.length} atoms...`);

    // Pre-compute all embeddings first
    const embeddings: { text: string; vector: Buffer }[] = [];
    for (const text of atoms) {
      const vector = await embedTextAsBuffer(text);
      embeddings.push({ text, vector });
    }

    // Batch insert all atoms in a single transaction for better performance
    const insertAll = db.transaction((items: { text: string; vector: Buffer }[]) => {
      for (const { text, vector } of items) {
        const { id } = saveAtom(text, vector);
        console.log(`> Atom ${id} vectorized.`);
      }
    });
    
    insertAll(embeddings);

    console.log('--- Ingestion Complete ---');
  } catch (error) {
    console.error('Ingestion failed:', error instanceof Error ? error.message : error);
    throw error;
  }
}