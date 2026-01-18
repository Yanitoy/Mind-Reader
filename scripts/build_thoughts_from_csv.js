const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const options = {};
for (let i = 0; i < args.length; i += 1) {
  if (args[i].startsWith("--")) {
    const key = args[i].slice(2);
    const value = args[i + 1];
    options[key] = value;
    i += 1;
  }
}

const inputPath = options.input || "backend/train.csv";
const outputPath = options.output || "frontend/public/thoughts.json";
const maxPerLabel = Number(options.maxPerLabel || 60);
const maxLength = Number(options.maxLength || 140);
const seed = Number(options.seed || 42);

const labelMap = {
  "0": "sad",
  "1": "happy",
  "2": "happy",
  "3": "angry",
  "4": "fear",
  "5": "surprise",
  "6": "neutral",
  "7": "disgust"
};

const textMap = {
  sadness: "sad",
  sad: "sad",
  anger: "angry",
  angry: "angry",
  fear: "fear",
  joy: "happy",
  happy: "happy",
  love: "happy",
  surprise: "surprise",
  disgust: "disgust",
  neutral: "neutral"
};

const emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
const buckets = Object.fromEntries(emotions.map((emotion) => [emotion, new Set()]));

function mulberry32(seedValue) {
  let t = seedValue + 0x6d2b79f5;
  return () => {
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(array, randomFn) {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = Math.floor(randomFn() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function normalizeLabel(raw) {
  if (raw === undefined || raw === null) {
    return null;
  }
  const value = String(raw).trim();
  if (!value) {
    return null;
  }
  if (Object.prototype.hasOwnProperty.call(labelMap, value)) {
    return labelMap[value];
  }
  const normalized = value.toLowerCase();
  return textMap[normalized] || null;
}

function parseLine(line) {
  const trimmed = line.trim();
  if (!trimmed) {
    return null;
  }
  const commaIndex = trimmed.indexOf(",");
  if (commaIndex === -1) {
    return null;
  }
  const text = trimmed.slice(0, commaIndex).trim();
  const label = trimmed.slice(commaIndex + 1).trim();
  return { text, label };
}

if (!fs.existsSync(inputPath)) {
  console.error(`CSV not found: ${inputPath}`);
  process.exit(1);
}

const lines = fs.readFileSync(inputPath, "utf8").split(/\r?\n/);
if (!lines.length) {
  console.error("CSV is empty.");
  process.exit(1);
}

const header = lines[0].split(",");
const hasHeader = header.includes("text") || header.includes("label");
const startIndex = hasHeader ? 1 : 0;

for (let i = startIndex; i < lines.length; i += 1) {
  const parsed = parseLine(lines[i]);
  if (!parsed) {
    continue;
  }
  const { text, label } = parsed;
  if (!text || text.length > maxLength) {
    continue;
  }
  const emotion = normalizeLabel(label);
  if (!emotion || !buckets[emotion]) {
    continue;
  }
  buckets[emotion].add(text);
}

const rng = mulberry32(seed);
const output = {};
emotions.forEach((emotion) => {
  const items = Array.from(buckets[emotion]);
  shuffle(items, rng);
  output[emotion] = items.slice(0, maxPerLabel);
});

const outputDir = path.dirname(outputPath);
fs.mkdirSync(outputDir, { recursive: true });
fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));

console.log(`Wrote ${outputPath}`);
