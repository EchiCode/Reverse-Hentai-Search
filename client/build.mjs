import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { build } from 'esbuild';
import { minify } from 'html-minifier-terser';

const outputDirectory = 'dist';
const textEntries = ['app.js', 'image-inference.js', 'search.js', 'search-release.js', 'search-quality.js', 'search-worker.js', 'centroid-simd.js',
    'centroid-worker-pool.js', 'centroid-rank-worker.js', 'styles.css'];

await rm(outputDirectory, { recursive: true, force: true });
await mkdir(outputDirectory);

await build({
    entryPoints: textEntries,
    outdir: outputDirectory,
    minify: true,
    target: ['es2022'],
    legalComments: 'none'
});

const html = await readFile('index.html', 'utf8');
await writeFile(`${outputDirectory}/index.html`, await minify(html, {
    collapseWhitespace: true,
    collapseBooleanAttributes: true,
    removeComments: true,
    removeRedundantAttributes: true,
    useShortDoctype: true
}));

await Promise.all([
    cp('assets', `${outputDirectory}/assets`, { recursive: true }),
    cp('fonts', `${outputDirectory}/fonts`, { recursive: true }),
    cp('centroids.lvq6.bin.gz', `${outputDirectory}/centroids.lvq6.bin.gz`),
    cp('centroid-simd.wasm', `${outputDirectory}/centroid-simd.wasm`),
    cp('_headers', `${outputDirectory}/_headers`)
]);
