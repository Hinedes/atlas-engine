import { pipeline, type FeatureExtractionPipeline } from '@xenova/transformers';

let embedderInstance: FeatureExtractionPipeline | null = null;

export async function getEmbedder(): Promise<FeatureExtractionPipeline> {
  if (embedderInstance) {
    return embedderInstance;
  }

  console.log('--- Initializing Quantized Semantic Core ---');
  embedderInstance = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
  }) as FeatureExtractionPipeline;
  console.log('Core Active.');

  return embedderInstance;
}

export async function embedText(text: string): Promise<Float32Array> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return output.data as Float32Array;
}

export function vectorToBuffer(vector: Float32Array): Buffer {
  return Buffer.from(vector.buffer);
}

export function bufferToVector(buffer: Buffer): Float32Array {
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
}
