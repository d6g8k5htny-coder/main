import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {quotedSourceLink,renderSourceQuote} from '../docs/site/source-quote.mjs';
import {documentFromHTML} from './fixtures/museum_fixture.mjs';
const source={repository:'d6g8k5htny-coder/Math-',commit:'a'.repeat(40),path:'notes/INDEX.md'};
const walk=(node,tag)=>(node.children??[]).flatMap(child=>[...(child.tagName===tag?[child]:[]),...walk(child,tag)]);
test('relative links resolve from the quoting source directory and commit, not the proof or live branch',()=>{
  assert.equal(quotedSourceLink('proof.md#scope',source),`https://github.com/${source.repository}/blob/${source.commit}/notes/proof.md#scope`);
  assert.equal(quotedSourceLink('a file.md',source),`https://github.com/${source.repository}/blob/${source.commit}/notes/a%20file.md`);
  assert.equal(quotedSourceLink('proof.md',{...source,repository:'d6g8k5htny-coder/main',path:'STATUS.md'}),`https://github.com/d6g8k5htny-coder/main/blob/${source.commit}/proof.md`);
});
test('quoted exact review comments and intentionally mutable source pointers keep their identities',()=>{
  for(const url of ['https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841270276','https://github.com/d6g8k5htny-coder/Math-/pull/129','https://github.com/d6g8k5htny-coder/Math-/blob/main/PROOF_INDEX.md']) assert.equal(quotedSourceLink(url,source),url);
});
test('unrecognized schemes, foreign destinations, encoded traversal and malformed identities stay literal',()=>{
  const targets=['javascript:alert(1)','data:text/html,x','http://github.com/d6g8k5htny-coder/main/issues/1','https://evil.test/a','https://github.com.evil.test/d6g8k5htny-coder/main/issues/1','https://github.com@evil.test/d6g8k5htny-coder/main/issues/1','https://github.com/d6g8k5htny-coder/private/blob/main/x','https://github.com/d6g8k5htny-coder/main/settings','https://github.com/d6g8k5htny-coder/main/blob/main/../x','https://github.com/d6g8k5htny-coder/main/blob/main/a?x=1','//evil.test/a','/proof.md','../proof.md','a/../proof.md','a//b','%2e%2e/proof.md','a\\b','a\nb','x\u202ey','a#','a#b#c'];
  for(const target of targets) assert.equal(quotedSourceLink(target,source),null,target);
  for(const bad of [{...source,commit:'main'},{...source,repository:'private'},{...source,path:'../INDEX.md'},null]) assert.equal(quotedSourceLink('proof.md',bad),null);
});
test('source HTML is always text and unsupported Markdown retains its literal spelling',()=>{
  const quote='<img src=x onerror=alert(1)> [unsafe](javascript:alert) ![image](proof.md) \\[escaped](proof.md) `literal [link](proof.md)` *single emphasis*';
  const [readable,original]=renderSourceQuote(documentFromHTML(),quote,source);
  assert.equal(walk(readable,'IMG').length,0);assert.equal(walk(readable,'A').length,0);
  assert.equal(readable.textContent,quote.replace('`literal [link](proof.md)`','literal [link](proof.md)'));
  assert.equal(original.children[1].textContent,quote);
});
test('every committed quote retains all wording and its original source independently of formatting',()=>{
  const manifest=JSON.parse(fs.readFileSync(new URL('../docs/site/museum.json',import.meta.url)));
  for(const [i,claim] of manifest.claims.entries()) {
    const origin=i<11?manifest.index_source:manifest.status_source;
    const [readable,disclosure]=renderSourceQuote(documentFromHTML(),claim.scope_quote,origin);
    assert.equal(disclosure.children[1].textContent,claim.scope_quote,claim.id);
    const expected=claim.scope_quote.replace(/`([^`\n]+)`/g,'$1').replace(/\*\*([^*\n]+)\*\*/g,'$1').replace(/\[([^\]\n]+)\]\(([^)\n]+)\)/g,(_all,label,target)=>{assert.ok(quotedSourceLink(target,origin),`${claim.id}: ${target}`);return label;});
    assert.equal(readable.textContent,expected,claim.id);
    assert.equal(walk(readable,'A').length,[...claim.scope_quote.matchAll(/\[([^\]\n]+)\]\(([^)\n]+)\)/g)].length,claim.id);
  }
});
