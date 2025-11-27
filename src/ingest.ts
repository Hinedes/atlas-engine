import fs from 'fs';
import path from 'path';
import db from './db';
import { embedTextAsBuffer, getEmbedder } from './embedder';
import type Database from 'better-sqlite3';

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

// Lazy initialization of prepared statements to avoid errors before schema exists
let insertAtomStmt: Database.Statement<[string]> | null = null;
let insertEmbeddingStmt: Database.Statement<[number | bigint, Buffer]> | null = null;

function getInsertAtomStmt(): Database.Statement<[string]> {
  if (!insertAtomStmt) {
    insertAtomStmt = db.prepare<[string]>(`INSERT INTO atoms (content) VALUES (?)`);
  }
  return insertAtomStmt;
}

function getInsertEmbeddingStmt(): Database.Statement<[number | bigint, Buffer]> {
  if (!insertEmbeddingStmt) {
    insertEmbeddingStmt = db.prepare<[number | bigint, Buffer]>(`INSERT INTO embeddings (atom_id, vector) VALUES (?, ?)`);
  }
  return insertEmbeddingStmt;
}

function saveAtom(content: string, vector: Buffer): AtomResult {
  const info = getInsertAtomStmt().run(content);
  getInsertEmbeddingStmt().run(info.lastInsertRowid, vector);

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