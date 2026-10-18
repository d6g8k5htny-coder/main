import importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('ep', root / 'tools/evidence_profile.py'); ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)
G = ep.load_gate()
def node(c, **f): return dict(classification=c, controlling=False, **f)
print('F5  family("GPT 5,Anthropic") =', ep.family('GPT 5,Anthropic'))
print('F5  family("OpenAI 4.7;not") =', ep.family('OpenAI 4.7;not'))
print('F12 family("OpenAI or ChatGPT") =', ep.family('OpenAI or ChatGPT'))
print('F13 family("OpenAI,ChatGPT") =', ep.family('OpenAI,ChatGPT'))
for a in ('not disclosed', 'possibly OpenAI', None):
    n = node('PROVED_REVIEWED', review_providers=['xAI']) if a is None else node('PROVED_REVIEWED', author_provider=a, review_providers=['xAI'])
    print('P2  author=%r reviewers=[xAI] ->' % a, ep.provider_distinct(n))
g = {'schema_version': 1, 'object': 'P', 'edges': [], 'nodes': {'math.y': node('PROVED_REVIEWED', author_provider='OpenAI', review_providers=['xAI'])}}
G.validate_graph_fail_closed(g)
print('L6  text label:', ep.tree_lines(G, g, 'math.y')[0])
