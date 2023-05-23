ckpt = 'e1499-s579000'
n_folds_per_length = 10  # number of folds per length
T = 1000

rule all:
    # input: expand(f'folds/{ckpt}-l{{nres}}-n{n_folds_per_length}.pt', nres=range(50, 128))
    input:
        expand(
            f'folds/gifs/{ckpt}-l{{nres}}/{{fold_id}}.gif',
            nres=range(50, 51), fold_id=range(n_folds_per_length)
        )

rule sample_folds:
    input: 'ckpts/{ckpt}.ckpt'
    output: 'folds/{ckpt}-l{nres}-n{n_folds_per_length}.pt'
    shell:
        'python -m foldingdiff_pytorch.sample '
        '--ckpt {T} '
        '--timepoints 1000 '
        '--num-residues {wildcards.nres} '
        '--batch-size {wildcards.n_folds_per_length} '
        '--output {output}'

rule fold_to_pdbs:
    input:
        f'folds/{{ckpt}}-l{{nres}}-n{n_folds_per_length}.pt'
    output:
        temp(expand( 
            'folds/pdbs/{{ckpt}}-l{{nres}}/{{fold_id}}/{{i}}.pdb',
            fold_id=range(n_folds_per_length), i=range(T) 
        ))
    params:
        outdir = lambda wc: f'folds/pdbs/{ckpt}-l{wc.nres}'
    shell:
        'python scripts/fold2pdbs.py '
        '-i {input} '
        '-o {params.outdir}'

rule pdb_to_png:
    input: 'folds/pdbs/{ckpt}-l{nres}/{fold_id}/{i}.pdb'
    output: temp('folds/pngs/{ckpt}-l{nres}/{fold_id}/{i}.png')
    conda: 'envs/pymol.yaml'
    shell: 'python scripts/pdb2png.py -i {input} -o {output}'

rule png_to_gif:
    input: expand('folds/pngs/{{ckpt}}-l{{nres}}/{{fold_id}}/{i}.png', i=range(T))
    output: 'folds/gifs/{ckpt}-l{nres}/{fold_id}.gif'
    shell: 'python scripts/pngs2gif.py -i {input} -o {output}'
