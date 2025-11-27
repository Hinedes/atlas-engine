import { pipeline, type FeatureExtractionPipeline } from '@xenova/transformers';

// Singleton pattern for embedder to avoid re-initialization
let embedderInstance: FeatureExtractionPipeline | null = null;
let initializationPromise: Promise<FeatureExtractionPipeline> | null = null;

/**
 * Gets or creates the embedder instance.
 * Uses singleton pattern to avoid expensive re-initialization.
 * Thread-safe through promise caching.
 */
export async function getEmbedder(): Promise<FeatureExtractionPipeline> {
  if (embedderInstance) {
    return embedderInstance;
  }

  // If initialization is already in progress, wait for it
  if (initializationPromise) {
    return initializationPromise;
  }

  // Start initialization
  initializationPromise = (async () => {
    console.log('--- Initializing Quantized Semantic Core ---');
    const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      quantized: true,
    });
    console.log('Core Active.');
    embedderInstance = embedder as FeatureExtractionPipeline;
    return embedderInstance;
  })();

  return initializationPromise;
}

/**
 * Embeds text into a vector representation.
 */
export async function embedText(text: string): Promise<Float32Array> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return output.data as Float32Array;
}

/**
 * Embeds text and returns as Buffer for storage.
 */
export async function embedTextAsBuffer(text: string): Promise<Buffer> {
  const vector = await embedText(text);
  return Buffer.from(vector.buffer);
}
