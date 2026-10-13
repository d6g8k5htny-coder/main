import sys
if len(sys.argv) == 3 and sys.argv[1] == '--mutant':
    if sys.argv[2] == 'M9':
        print('unknown mutant'); sys.exit(2)
    raise RuntimeError('synthetic mutant crash')
print('{"fixture": true}')
