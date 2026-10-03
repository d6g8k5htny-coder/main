import {wireReferenceForm} from './source-reference.mjs?site-release=99e6a33de734525f28b18c3c89b3fe85717576e2b0a5d58989cb2b189ac40bda';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), output:byId('reference-output'), link:byId('reference-link'), status:byId('reference-status')});
byId('reference-builder').hidden = false;
