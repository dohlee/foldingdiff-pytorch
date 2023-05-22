ckpt = 'ckpts/e1499-s579000'
bsz = 10  # number of folds per length

rule all:
    input: expand(f'folds/{ckpt}-l{{nres}}-n{bsz}.pt', nres=range(50, 128))

rule sample_folds:
    input: '{ckpt}.ckpt'
    output: 'folds/{ckpt}-l{nres}-n{bsz}.pt'
    shell:
        'python -m foldingdiff_pytorch.sample '
        '--ckpt {input} '
        '--timepoints 1000 '
        '--num-residues {wildcards.nres} '
        '--batch-size {wildcards.bsz} '
        '--output {output}'
