import fs from 'fs';
import path from 'path';
import db from './db';
import { embedText, vectorToBuffer } from './embedder';

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

function saveAtom(content: string, vector: Buffer): AtomResult {
  const insertAtom = db.prepare(`INSERT INTO atoms (content) VALUES (?)`);
  const insertEmbedding = db.prepare(`INSERT INTO embeddings (atom_id, vector) VALUES (?, ?)`);

  const info = insertAtom.run(content);
  insertEmbedding.run(info.lastInsertRowid, vector);

  return { id: info.lastInsertRowid, content };
}

export async function ingest(filename = 'briefing.md'): Promise<void> {
  try {
    const atoms = loadDocument(filename);

    console.log(`Processing ${atoms.length} atoms...`);

    for (const text of atoms) {
      const vector = await embedText(text);
      const { id } = saveAtom(text, vectorToBuffer(vector));
      console.log(`> Atom ${id} vectorized.`);
    }

    console.log('--- Ingestion Complete ---');
  } catch (error) {
    console.error('Ingestion failed:', error instanceof Error ? error.message : error);
    throw error;
  }
}