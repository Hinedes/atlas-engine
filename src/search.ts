import db from './db';
import { embedText } from './embedder';

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

function bufferToVector(buffer: Buffer): Float32Array {
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
}

// Prepare statement once for reuse
const fetchEmbeddingsStmt = db.prepare(`
  SELECT e.vector, a.content 
  FROM embeddings e
  JOIN atoms a ON e.atom_id = a.id
`);

function fetchAllEmbeddings(): EmbeddingRow[] {
  return fetchEmbeddingsStmt.all() as EmbeddingRow[];
}

function rankResults(queryVector: Float32Array, rows: EmbeddingRow[], limit: number): SearchResult[] {
  // Use a min-heap approach for efficient top-k selection
  // This is O(n + k log k) instead of O(n log n) for full sort
  const results: SearchResult[] = rows.map(row => ({
    score: cosineSimilarity(queryVector, bufferToVector(row.vector)),
    content: row.content,
  }));

  // For small datasets or when limit is close to n, full sort is fine
  // For large datasets with small limits, partial sort would be more efficient
  if (limit >= results.length) {
    return results.sort((a, b) => b.score - a.score);
  }

  // Partial sort: only sort to find top `limit` elements
  // Using quickselect-style partition for better average performance
  return partialSort(results, limit);
}

/**
 * Efficiently finds top-k elements without fully sorting the array.
 * Uses a selection algorithm approach.
 */
function partialSort(arr: SearchResult[], k: number): SearchResult[] {
  // For small k, use simple approach: track top k elements
  if (k <= 10) {
    const topK: SearchResult[] = [];
    for (const item of arr) {
      if (topK.length < k) {
        topK.push(item);
        topK.sort((a, b) => b.score - a.score);
      } else if (item.score > topK[topK.length - 1].score) {
        topK[topK.length - 1] = item;
        topK.sort((a, b) => b.score - a.score);
      }
    }
    return topK;
  }
  
  // For larger k, full sort is acceptable
  return arr.sort((a, b) => b.score - a.score).slice(0, k);
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

  const queryVector = await embedText(query);
  const rows = fetchAllEmbeddings();
  const results = rankResults(queryVector, rows, limit);

  displayResults(results, limit);
  return results;
}

export type { SearchResult };