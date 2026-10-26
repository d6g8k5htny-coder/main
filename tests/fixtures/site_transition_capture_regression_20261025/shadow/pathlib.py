with open('.import-shadow-marker', 'a') as stream:
    stream.write('SHADOW_IMPORT\n')
raise RuntimeError('unqualified shadow imported')
