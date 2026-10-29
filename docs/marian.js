// Marian translation models (FuguMT, OPUS-MT) exported to ONNX: their SentencePiece tokenizer, and greedy
// decoding with an encoder and a decoder without a KV cache. Pure JS that takes ort as a parameter, like
// ocr-core.js, so the Node test runs the code marian-worker.js does.

// tokenizer.json (tools/prepare_models.py): the source language's SentencePiece unigram pieces with their
// scores, the vocabulary shared by both sides (id -> token), and the special ids.
export function tokenizer({ pieces, vocab, unk, eos, pad, start }) {
  const score = new Map(pieces);
  const ids = new Map(vocab.map((t, i) => [t, i]));
  const longest = Math.max(...pieces.map(([p]) => [...p].length));
  const unkScore = Math.min(...pieces.map(([, s]) => s)) - 10; // SentencePiece's penalty for an unknown character
  return {
    unk, eos, pad, start,
    // Text -> ids, ending with </s>. SentencePiece's nmt_nfkc normaliser is NFKC with whitespace runs
    // squeezed to one space; spaces, and the start, are marked ▁. Then the most probable split into
    // pieces (Viterbi), an unknown character on its own (consecutive ones merged, as SentencePiece does).
    encode(text) {
      const s = [..."▁" + text.normalize("NFKC").replace(/\s+/gu, " ").trim().replaceAll(" ", "▁")];
      const best = new Float64Array(s.length + 1).fill(-Infinity), from = new Int32Array(s.length + 1);
      best[0] = 0;
      for (let i = 0; i < s.length; i++) {
        if (best[i] === -Infinity) continue;
        let piece = "", single = false;
        for (let j = i; j < Math.min(s.length, i + longest); j++) {
          piece += s[j];
          const sc = score.get(piece);
          if (sc === undefined) continue;
          if (j === i) single = true;
          if (best[i] + sc > best[j + 1]) { best[j + 1] = best[i] + sc; from[j + 1] = i; }
        }
        if (!single && best[i] + unkScore > best[i + 1]) { best[i + 1] = best[i] + unkScore; from[i + 1] = i; }
      }
      const split = [];
      for (let j = s.length; j > 0; j = from[j]) split.push(s.slice(from[j], j).join(""));
      // Unknown pieces keep their text, and Marian looks that up in its vocabulary too (蕨 is in
      // OPUS-MT zh-en's, though not among its source pieces).
      const merged = [];
      for (const p of split.reverse()) {
        if (!score.has(p) && merged.length && !score.has(merged.at(-1))) merged[merged.length - 1] += p;
        else merged.push(p);
      }
      return [...merged.map((p) => ids.get(p) ?? unk), eos];
    },
    decode: (out) => out.filter((i) => i !== eos && i !== pad && i !== unk).map((i) => vocab[i]).join("").replaceAll("▁", " ").trim(),
  };
}

// One sentence: encode it once, then add the likeliest next token until </s>. Without a KV cache the
// decoder reads all the tokens so far each step, fine for a sentence of a few dozen.
async function sentence(ort, { encoder, decoder, tok }, text) {
  const ids = tok.encode(text);
  const i64 = (a) => new ort.Tensor("int64", BigInt64Array.from(a, BigInt), [1, a.length]);
  const mask = i64(ids.map(() => 1));
  const { last_hidden_state: hidden } = await encoder.run({ input_ids: i64(ids), attention_mask: mask });
  const out = [tok.start];
  while (out.length <= Math.min(256, 2 * ids.length + 10)) {
    const { logits } = await decoder.run({ input_ids: i64(out), encoder_attention_mask: mask, encoder_hidden_states: hidden }, ["logits"]);
    const V = logits.dims[2], last = logits.data.subarray((out.length - 1) * V, out.length * V);
    let next = tok.pad === 0 ? 1 : 0;
    for (let v = 0; v < V; v++) if (v !== tok.pad && last[v] > last[next]) next = v;
    if (next === tok.eos) break;
    out.push(next);
  }
  return tok.decode(out.slice(1));
}

// A paragraph, a sentence at a time (these models are trained on single sentences): split after
// 。！？!? and after a full stop followed by a space.
export async function translate(ort, model, text) {
  const parts = text.split(/(?<=[。！？!?]|\.(?=\s))\s*/u).filter((s) => s.trim());
  const out = [];
  for (const s of parts) out.push(await sentence(ort, model, s));
  return out.join(" ");
}
