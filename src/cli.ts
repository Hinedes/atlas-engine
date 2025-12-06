#!/usr/bin/env node
import { Command } from 'commander';
import { initSchema } from './schema';
import { ingest } from './ingest';
import { search } from './search';

const program = new Command();

program
  .name('atlas')
  .description('Atlas Engine - Semantic Knowledge Management System')
  .version('1.0.0');

program
  .command('init')
  .description('Initialize the database schema')
  .action(() => {
    initSchema();
    console.log('Atlas Engine: Online.');
  });

program
  .command('ingest')
  .description('Ingest all documents from the vault into the knowledge base')
  .action(async () => {
    initSchema(); // Ensure schema exists
    await ingest();
  });

program
  .command('search <query>')
  .description('Search the knowledge base')
  .option('-l, --limit <number>', 'Number of results to return', '3')
  .action(async (query: string, options: { limit: string }) => {
    const limit = parseInt(options.limit, 10);
    await search(query, limit);
  });

program.parse();
