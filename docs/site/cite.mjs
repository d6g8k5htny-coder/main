import {wireReferenceForm} from './source-reference.mjs?site-release=6c905b8133e26b3a9feb1ec74bbcc429c3146ea712b475f8f4b119376e22ec8b';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), output:byId('reference-output'), link:byId('reference-link'), status:byId('reference-status')});
byId('reference-builder').hidden = false;
