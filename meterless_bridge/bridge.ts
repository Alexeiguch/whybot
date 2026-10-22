// JSON-lines RPC bridge. Python writes {"id","method","params"} per line on stdin
// and reads {"id","result"|"error"} per line on stdout. The engines are the
// Meterless TypeScript reference implementations vendored under ../vendor/meterless
// (Apache-2.0, upstream github.com/Meterless/Meterless @ 1742dec).
import * as readline from "node:readline";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const enginesDir = process.env.METERLESS_ENGINES_DIR ?? path.join(here, "..", "vendor", "meterless", "engines");
const { HMEM } = await import(path.join(enginesDir, "hmem/reference/src/index.ts"));
const { WorldModel } = await import(path.join(enginesDir, "world-model/reference/src/index.ts"));
const { Markovian, CarryoverService, parseMarkers } = await import(path.join(enginesDir, "markovian/reference/src/index.ts"));

const dataDir = process.env.METERLESS_DATA_DIR ?? path.join(here, "..", ".meterless");
const hmem = new HMEM({ persistDir: `${dataDir}/hmem`, actor: "whybot" });
const world = new WorldModel({ storage: "local", namespace: "whybot-evidence", directory: `${dataDir}/world` });
const carry = new CarryoverService();
const strip = (r: any) => (r && r.embedding ? { ...r, embedding: `[${r.embedding.length}-dim]` } : r);

const methods: Record<string, (p: any) => unknown> = {
  "ping": () => ({ ok: true, enginesDir, dataDir }),
  "hmem.add": (p) => strip(hmem.add(p)),
  "hmem.query": (p) => {
    const r: any = hmem.query(p.query, p.opts);
    return { ...r, memories: (r.memories ?? []).map((x: any) => ({ relevance: x.relevance, memory: strip(x.memory) })) };
  },
  "hmem.feedback": (p) => { hmem.feedback(p.id, p.kind); return { ok: true }; },
  "wm.upsertEntity": (p) => world.upsertEntity(p),
  "wm.assertFact": (p) => world.assertFact(p),
  "wm.relate": (p) => world.relate(p),
  "wm.about": (p) => world.view("facts").about(p.id),
  "wm.auditForRun": (p) => world.audit().filter((a: any) => a.runId === p.runId),
  // Incremental carryover: one investigation iteration = one Markovian chunk.
  "markovian.carry": async (p) => {
    const parsed = parseMarkers(p.chunkText);
    const config = { chunkSize: 8000, maxChunks: 3, carryoverTokens: p.carryoverTokens ?? 512, overlapTokens: 0 };
    const r = await carry.next({ markerState: parsed.state, previousCarryover: p.previousCarryover ?? "", chunkText: p.chunkText, config });
    return { carryover: r.carryover, level: r.level, done: parsed.done, chunkTokens: Math.ceil(p.chunkText.length / 4), carryTokens: Math.ceil(r.carryover.length / 4) };
  },
  // Records the finished investigation as a Markovian run (history + efficiency stats).
  "markovian.record": async (p) => {
    const m = new Markovian({ chunkConfig: { chunkSize: 8000, maxChunks: Math.max(1, p.steps.length), carryoverTokens: p.carryoverTokens ?? 512, overlapTokens: 0 },
                              historyDir: `${dataDir}/markovian` });
    const run = await m.run({ goal: p.goal, reflect: false, stepFn: ({ step }: any) => ({ content: p.steps[step] ?? "[TASK_COMPLETE]" }) });
    return { id: run.id, status: run.status, chunks: run.chunks.length, used: run.totalTokensUsed, saved: run.totalTokensSaved, efficiencyPercent: run.efficiencyPercent };
  },
};

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  let msg: any;
  try { msg = JSON.parse(line); } catch { return; }
  try {
    const fn = methods[msg.method];
    if (!fn) throw new Error(`unknown method ${msg.method}`);
    const result = await fn(msg.params ?? {});
    process.stdout.write(JSON.stringify({ id: msg.id, result }) + "\n");
  } catch (e) {
    process.stdout.write(JSON.stringify({ id: msg.id, error: String((e as Error)?.message ?? e) }) + "\n");
  }
});
process.stdout.write(JSON.stringify({ id: 0, result: { ready: true } }) + "\n");
