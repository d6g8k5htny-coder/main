import {wireReferenceForm} from './source-reference.mjs?site-release=dc3b8a3d50e06b183520769b998318b774231b7fc56eda438e4a4878b963d1a3';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), output:byId('reference-output'), link:byId('reference-link'), status:byId('reference-status')});
byId('reference-builder').hidden = false;
