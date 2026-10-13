// Controls: (i) dropping pointer_only on the other five pointer cards is refused (only d5-pin slips through, see p3a).
import {realManifest, validate} from './lib.mjs';
for (const i of [0, 1, 2, 9, 11, 12]) {
  const m = realManifest(); const r = m.claims[i].review; delete r.pointer_only; delete r.review_url; delete r.review_notice;
  console.log(`${m.claims[i].id}: pointer_only removed -> ${validate(m)}`);
}
