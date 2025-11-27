import db from './db';
import { pipeline, type FeatureExtractionPipeline } from '@xenova/transformers';

interface SearchResult {
  score: number;
  content: string;
}

interface EmbeddingRow {
  vector: Buffer;
  content: string;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function initializeEmbedder(): Promise<FeatureExtractionPipeline> {
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  });
  return embedder as FeatureExtractionPipeline;
}

async function embedQuery(embedder: FeatureExtractionPipeline, query: string): Promise<Float32Array> {
  const output = await embedder(query, { pooling: 'mean', normalize: true });
  return output.data as Float32Array;
}

function fetchAllEmbeddings(): EmbeddingRow[] {
  const stmt = db.prepare(`
    SELECT e.vector, a.content 
    FROM embeddings e
    JOIN atoms a ON e.atom_id = a.id
  `);
  return stmt.all() as EmbeddingRow[];
}

function bufferToVector(buffer: Buffer): Float32Array {
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
}

function rankResults(queryVector: Float32Array, rows: EmbeddingRow[]): SearchResult[] {
  return rows
    .map(row => ({
      score: cosineSimilarity(queryVector, bufferToVector(row.vector)),
      content: row.content,
    }))
    .sort((a, b) => b.score - a.score);
}

function displayResults(results: SearchResult[], limit = 3): void {
  console.log('--- TARGETS ACQUIRED ---');
  results.slice(0, limit).forEach(r => {
    const snippet = r.content.split('\n')[0].substring(0, 60);
    console.log(`[${(r.score * 100).toFixed(1)}% Match] ${snippet}...`);
  });
}

export async function search(query: string, limit = 3): Promise<SearchResult[]> {
  console.log(`\n🔍 Searching Neural Memory for: "${query}"`);

  const embedder = await initializeEmbedder();
  const queryVector = await embedQuery(embedder, query);
  const rows = fetchAllEmbeddings();
  const results = rankResults(queryVector, rows);

  displayResults(results, limit);
  return results;
}

export type { SearchResult };