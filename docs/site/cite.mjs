import {wireReferenceForm} from './source-reference.mjs?site-release=afac1358a71a41317dc244b90babf1c60be46544a66c800c47b1111e4f3dce37';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), output:byId('reference-output'), link:byId('reference-link'), status:byId('reference-status')});
byId('reference-builder').hidden = false;
