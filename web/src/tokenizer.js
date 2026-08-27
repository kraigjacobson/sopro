const TERMINALS = new Set(['.', '!', '?', '-', ',', ';', ':']);
const LANGUAGE_TAGS = { en: '<|lang_en|>', pt: '<|lang_pt|>', fr: '<|lang_fr|>', de: '<|lang_de|>' };

function languageTag(language) {
  const code = String(language ?? '').trim().toLowerCase();
  if (!code || code === 'auto') return null;
  const tag = LANGUAGE_TAGS[code];
  if (!tag) throw new Error(`Unsupported language: ${language}`);
  return tag;
}

export function normalizeText(value, language = null) {
  let text = String(value ?? '').trim();
  const tag = languageTag(language);
  if (!text && tag) return tag;
  if (!text) text = 'You need to add some text for me to talk.';
  if (text[0] === text[0].toLocaleLowerCase() && text[0] !== text[0].toLocaleUpperCase()) {
    text = text[0].toLocaleUpperCase() + text.slice(1);
  }
  text = text.replaceAll('…', '...').replace(/[“”]/g, '"').replace(/[‘’]/g, "'");
  text = text.replace(/\s+([,.!?;:])/g, '$1').replace(/\s+/g, ' ').trim();
  if (![...TERMINALS].some((ending) => text.endsWith(ending))) text += '.';
  if (tag) text = `${tag} ${text}`;
  return text;
}


const SENTENCE_END = /(?<=[.!?\u2026])\s+/;
const CLAUSE_END = /(?<=[,;:])\s+/;

function pack(parts, maxChars) {
  const out = [];
  let current = '';
  for (const part of parts) {
    if (!current) current = part;
    else if (current.length + 1 + part.length <= maxChars) current = `${current} ${part}`;
    else { out.push(current); current = part; }
  }
  if (current) out.push(current);
  return out;
}

export function splitText(value, maxChars) {
  const text = String(value).split(/\s+/).filter(Boolean).join(' ');
  if (text.length <= maxChars) return text ? [text] : [];
  const segments = [];
  for (const sentence of pack(text.split(SENTENCE_END), maxChars)) {
    if (sentence.length <= maxChars) { segments.push(sentence); continue; }
    for (const clause of pack(sentence.split(CLAUSE_END), maxChars)) {
      if (clause.length <= maxChars) segments.push(clause);
      else segments.push(...pack(clause.split(' '), maxChars));
    }
  }
  return segments;
}

function trieNode() { return { next: new Map(), id: -1, score: -Infinity }; }

export class SentencePieceTokenizer {
  constructor(spec) {
    if (spec.format !== 'sentencepiece-unigram-v1') throw new Error(`Unsupported tokenizer format: ${spec.format}`);
    this.spec = spec;
    this.root = trieNode();
    this.byteIds = new Map();
    for (let id = 0; id < spec.pieces.length; id++) {
      if (spec.types[id] === 'byte') {
        this.byteIds.set(Number.parseInt(spec.pieces[id].slice(3, 5), 16), id);
        continue;
      }
      if (spec.types[id] !== 'normal') continue;
      let node = this.root;
      for (const char of Array.from(spec.pieces[id])) {
        if (!node.next.has(char)) node.next.set(char, trieNode());
        node = node.next.get(char);
      }
      node.id = id;
      node.score = spec.scores[id];
    }
  }

  encode(value, language = null) {
    const normalized = normalizeText(value, language).normalize('NFKC');
    const chars = Array.from(`▁${normalized.replace(/\s+/g, '▁')}`);
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const back = Array(n + 1).fill(null);
    best[0] = 0;
    for (let start = 0; start < n; start++) {
      if (!Number.isFinite(best[start])) continue;
      let node = this.root;
      let matched = false;
      for (let end = start; end < n; end++) {
        node = node.next.get(chars[end]);
        if (!node) break;
        if (node.id >= 0) {
          matched = true;
          const score = best[start] + node.score;
          if (score > best[end + 1]) {
            best[end + 1] = score;
            back[end + 1] = { start, ids: [node.id] };
          }
        }
      }
      if (!matched || !Number.isFinite(best[start + 1])) {
        const bytes = new TextEncoder().encode(chars[start]);
        const ids = Array.from(bytes, (byte) => this.byteIds.get(byte) ?? this.spec.unkId);
        const score = best[start] - 100 - ids.length;
        if (score > best[start + 1]) {
          best[start + 1] = score;
          back[start + 1] = { start, ids };
        }
      }
    }
    const pieces = [];
    for (let cursor = n; cursor > 0;) {
      const item = back[cursor];
      if (!item) { pieces.unshift(this.spec.unkId); cursor -= 1; continue; }
      pieces.unshift(...item.ids);
      cursor = item.start;
    }
    return [this.spec.bosId, ...pieces, this.spec.eosId].slice(0, this.spec.maxLength);
  }
}
