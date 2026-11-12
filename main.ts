// Predict Roblox's Up-and-Coming top 10 — place-by-place probability forecast.
//
// The chart ranks games published in the last 28 days by recent user growth, so a game's
// future place is driven by its current rank, how fast its rank has been moving, and the
// hazard of dropping out of the top 10. This program fits those effects from the history it
// is given, at run time, and writes one forecast for every hourly read of the scored window:
//   - per horizon bucket (6h) it fits P(place | rank) as a log-normal kernel whose shift and
//     width maximise the observed transitions, plus the escape probability per rank bucket;
//   - each game's kernel is weighted by the probability it stays inside the top 10;
//   - the per-place kernels are summed over the 72 hours and normalised into per-place
//     distributions, leaving a small residual that the chart's unnamed games share.
// Node 22, standard library only, deterministic: the same history always writes the same file.
//
//   MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const PLACES = 10;
const NBUCK = 12;              // 12 horizon buckets of 6 hours, covering hours 1..72
const CAP = 30;                // games deeper than this at the end of the history are not named
const RESIDUAL = 0.02;         // share of every place left to the games not named
const BETA = 0.0;              // rank-velocity weight
const TAU = 18;                // hours the velocity keeps acting
const HOURS = Array.from({ length: 72 }, (_, i) => i + 1);

interface Row { rank: number; universe_id: number }
interface Snapshot { at: number; rank: Map<number, number> }

function rankBucket(r: number): number {
  if (r <= 1) return 0;
  if (r <= 2) return 1;
  if (r <= 3) return 2;
  if (r <= 5) return 3;
  if (r <= 8) return 4;
  if (r <= 12) return 5;
  if (r <= 20) return 6;
  if (r <= 30) return 7;
  return 8;
}

function bucketOf(h: number): number {
  return Math.min(Math.floor((h - 1) / 6), NBUCK - 1);
}

function kernel(k: number, mu: number, sd: number): number {
  const z = (Math.log(k) - mu) / sd;
  return Math.exp(-0.5 * z * z) / k;
}

function loadHistory(path: string): Snapshot[] {
  const snaps: Snapshot[] = [];
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line) continue;
    let d: any;
    try { d = JSON.parse(line); } catch { continue; }
    const rows: Row[] = (d.up_and_coming ?? []).filter((r: Row) => r && r.universe_id);
    if (rows.length === 0) continue;
    rows.sort((a, b) => a.rank - b.rank);
    const rank = new Map<number, number>();
    for (const r of rows) rank.set(Number(r.universe_id), Number(r.rank));
    snaps.push({ at: Date.parse(d.at), rank });
  }
  snaps.sort((a, b) => a.at - b.at);
  return snaps;
}

// counts[bucket][rank][place]: place 0 means the game had left the top 10 by then
function buildCounts(snaps: Snapshot[]): number[][][][] {
  const counts: number[][][][] = Array.from({ length: NBUCK }, () =>
    Array.from({ length: CAP + 1 }, () => new Array<number>(PLACES + 1).fill(0)));
  for (let i = 0; i < snaps.length - 1; i++) {
    const a = snaps[i]!;
    for (let j = i + 1; j < snaps.length; j++) {
      const b = snaps[j]!;
      const h = (b.at - a.at) / 3.6e6;
      if (h > 72) break;
      const bk = bucketOf(h);
      for (const [g, r0] of a.rank) {
        if (r0 > CAP) continue;
        const rb = b.rank.get(g);
        const place = rb !== undefined && rb <= PLACES ? rb : 0;
        counts[bk]![r0]![place] += 1;
      }
    }
  }
  return counts;
}

function fitTable(counts: number[][][][]): { shift: number[]; sd: number[]; esc: number[][] } {
  const shift = new Array<number>(NBUCK).fill(0);
  const sd = new Array<number>(NBUCK).fill(0.6);
  const esc: number[][] = Array.from({ length: NBUCK }, () => new Array<number>(9).fill(0.5));
  for (let bk = 0; bk < NBUCK; bk++) {
    const escN = new Array<number>(9).fill(0);
    const escD = new Array<number>(9).fill(0);
    const rowTotals = new Array<number>(CAP + 1).fill(0);
    for (let r0 = 1; r0 <= CAP; r0++) {
      const row = counts[bk]![r0]!;
      let total = 0;
      for (let k = 0; k <= PLACES; k++) total += row[k]!;
      rowTotals[r0] = total;
      if (total === 0) continue;
      const rb = rankBucket(r0);
      escN[rb] += row[0]!;
      escD[rb] += total;
    }
    for (let rb = 0; rb < 9; rb++) esc[bk]![rb] = (escN[rb]! + 1.5) / (escD[rb]! + 3);

    let bestLL = -Infinity;
    let bestShift = 0;
    let bestSd = 0.6;
    for (let si = -12; si <= 12; si++) {
      const sh = si * 0.1;
      for (let sdi = 3; sdi <= 20; sdi++) {
        const s = sdi * 0.1;
        let ll = 0;
        for (let r0 = 1; r0 <= CAP; r0++) {
          if (rowTotals[r0] === 0) continue;
          const mu = Math.log(r0) + sh;
          let Z = 0;
          const ks = new Array<number>(PLACES + 1).fill(0);
          for (let k = 1; k <= PLACES; k++) { ks[k] = kernel(k, mu, s); Z += ks[k]!; }
          if (Z <= 0) { ll = -Infinity; break; }
          const row = counts[bk]![r0]!;
          for (let k = 1; k <= PLACES; k++) if (row[k]! > 0) ll += row[k]! * Math.log(ks[k]! / Z);
        }
        if (ll > bestLL) { bestLL = ll; bestShift = sh; bestSd = s; }
      }
    }
    if (Number.isFinite(bestLL)) { shift[bk] = bestShift; sd[bk] = bestSd; }
    else if (bk > 0) { shift[bk] = shift[bk - 1]!; sd[bk] = sd[bk - 1]!; }
  }
  return { shift, sd, esc };
}

function velocities(snaps: Snapshot[], lookbackH = 24): Map<number, number> {
  const last = snaps[snaps.length - 1]!;
  const series = new Map<number, Array<[number, number]>>();
  for (const r of snaps) {
    const off = (r.at - last.at) / 3.6e6;
    if (off > lookbackH) continue;
    for (const [g, rk] of r.rank) {
      if (!last.rank.has(g)) continue;
      const s = series.get(g) ?? [];
      s.push([off, rk]);
      series.set(g, s);
    }
  }
  const out = new Map<number, number>();
  for (const [g, pts] of series) {
    const prior = pts.filter((p) => p[0] < -0.4).sort((a, b) => a[0] - b[0]);
    let vel = 0;
    if (prior.length >= 2) {
      const [t0, ra] = prior[0]!;
      const [t1, rb] = prior[prior.length - 1]!;
      if (t1 - t0 > 0.4) vel = (Math.log(rb) - Math.log(ra)) / (t1 - t0);
    }
    out.set(g, Math.max(-0.1, Math.min(0.1, vel)));
  }
  return out;
}

function main(): void {
  const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as { output: string };
  const snaps = loadHistory(join(INPUT_DIR, 'history.ndjson'));
  if (snaps.length === 0) throw new Error('empty history');
  const counts = buildCounts(snaps);
  const { shift, sd, esc } = fitTable(counts);
  const vel = velocities(snaps);
  const last = snaps[snaps.length - 1]!;

  const agg: Array<Map<number, number>> = Array.from({ length: PLACES + 1 }, () => new Map<number, number>());
  const named: number[] = [];
  for (const [g, r0] of last.rank) {
    if (r0 > CAP) continue;
    named.push(g);
    const v = vel.get(g) ?? 0;
    const rb = rankBucket(r0);
    for (const h of HOURS) {
      const bk = bucketOf(h);
      const mu = Math.log(r0) + shift[bk]! + BETA * v * Math.min(h, TAU);
      const s = sd[bk]!;
      let Z = 0;
      const ks = new Array<number>(PLACES + 1).fill(0);
      for (let k = 1; k <= PLACES; k++) { ks[k] = kernel(k, mu, s); Z += ks[k]!; }
      if (Z <= 0) continue;
      const keep = 1 - esc[bk]![rb]!;
      for (let k = 1; k <= PLACES; k++) {
        agg[k]!.set(g, (agg[k]!.get(g) ?? 0) + (keep * ks[k]!) / Z);
      }
    }
  }

  const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
  for (let k = 1; k <= PLACES; k++) {
    let total = 0;
    for (const v of agg[k]!.values()) total += v;
    if (total <= 0) continue;
    const scale = (1 - RESIDUAL) / total;
    for (const [g, v] of [...agg[k]!.entries()].sort((a, b) => a[0] - b[0])) {
      const p = Math.round(v * scale * 1e6) / 1e6;
      if (p >= 1e-6) probabilities.push({ universe_id: g, place: k, p });
    }
  }

  const top10: number[] = [];
  const used = new Set<number>();
  for (let k = 1; k <= PLACES; k++) {
    let best = -1;
    let bestP = -1;
    for (const [g, v] of agg[k]!) {
      if (used.has(g)) continue;
      if (v > bestP) { bestP = v; best = g; }
    }
    if (best >= 0) { top10.push(best); used.add(best); }
  }
  for (const g of named.sort((a, b) => last.rank.get(a)! - last.rank.get(b)!)) {
    if (top10.length >= PLACES) break;
    if (!used.has(g)) { top10.push(g); used.add(g); }
  }

  writeFileSync(join(OUTPUT_DIR, task.output), JSON.stringify({ probabilities, top10 }) + '\n');
}

main();
