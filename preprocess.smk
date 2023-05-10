pdbs = glob_wildcards('data/dompdb/{pdb}.pdb').pdb

rule all:
    input: expand('data/npy/{pdb}.npy', pdb=pdbs)

rule pdb2npy:
    input: 'data/dompdb/{pdb}.pdb'
    output: 'data/npy/{pdb}.npy'
    shell: 'python scripts/pdb2npy.py -i {input} -o {output}'
