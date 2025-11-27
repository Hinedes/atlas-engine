import fs from 'fs';
import path from 'path';
import db from './db';
import { pipeline, type FeatureExtractionPipeline } from '@xenova/transformers';

const VAULT_DIR = path.join(process.cwd(), 'vault');

interface AtomResult {
  id: number | bigint;
  content: string;
}

async function initializeEmbedder(): Promise<FeatureExtractionPipeline> {
  console.log('--- Initializing Quantized Semantic Core ---');
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  });
  console.log('Core Active.');
  return embedder as FeatureExtractionPipeline;
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

async function embedText(embedder: FeatureExtractionPipeline, text: string): Promise<Buffer> {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Buffer.from((output.data as Float32Array).buffer);
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
    const embedder = await initializeEmbedder();
    const atoms = loadDocument(filename);

    console.log(`Processing ${atoms.length} atoms...`);

    for (const text of atoms) {
      const vector = await embedText(embedder, text);
      const { id } = saveAtom(text, vector);
      console.log(`> Atom ${id} vectorized.`);
    }

    console.log('--- Ingestion Complete ---');
  } catch (error) {
    console.error('Ingestion failed:', error instanceof Error ? error.message : error);
    throw error;
  }
}