import {wireReferenceForm} from './source-reference.mjs?site-release=653895fa4a65723a5495fafad9a91b12ddeb3f656b46d57a81272f46901e70c7';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), output:byId('reference-output'), link:byId('reference-link'), status:byId('reference-status')});
byId('reference-builder').hidden = false;
