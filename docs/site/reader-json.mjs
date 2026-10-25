// Shared read-only decoding; duplicate raw keys never overwrite evidence.
const MAX_BYTES=32*1024*1024;
const fail=message=>{throw Error(message);};
// JSON.parse validates grammar, then an iterative scan checks the original text.
// Keys are decoded individually so escaped spellings share the same identity.
// No reserialization can erase a duplicate before this check sees its bytes.
export function parseStrictJSON(text) {
  if (typeof text !== "string" || text.length > MAX_BYTES)
    fail("JSON text exceeds capacity.");
  const data = JSON.parse(text);
  const stack = [];
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (character === "{" || character === "[") {
      stack.push(character === "{" ? new Set() : null);
      if (stack.length > 128) fail("JSON nesting depth exceeds capacity.");
    } else if (character === "}" || character === "]") {
      stack.pop();
    } else if (character === '"') {
      const start = index;
      index++;
      while (text[index] !== '"') {
        if (text[index] === "\\") index++;
        index++;
      }
      let next = index + 1;
      while (/\s/.test(text[next] || "") && next < text.length) next++;
      if (text[next] === ":") {
        const key = JSON.parse(text.slice(start, index + 1));
        const keys = stack.at(-1);
        if (keys.has(key)) fail("Duplicate JSON key: " + key);
        keys.add(key);
      }
    }
  }
  return data;
}
