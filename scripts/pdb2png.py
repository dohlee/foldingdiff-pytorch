import argparse
import pymol
import os

from tqdm import tqdm


def pdb2png(pdb, file):
    """Convert the pdb file into a png, returns output filename"""
    # https://gist.github.com/bougui505/11401240
    pymol.cmd.load(pdb)
    pymol.cmd.show("cartoon")
    pymol.cmd.color("palegreen")
    pymol.cmd.bg_color("white")
    pymol.cmd.set("ray_opaque_background", 1)
    pymol.cmd.set("ray_trace_mode", 1)
    pymol.cmd.set("ray_trace_color", "black")
    pymol.cmd.set("ray_trace_gain", 0.01)
    #     pymol.cmd.origin(position=[0.0, 0.0, 0.0])
    #     pymol.cmd.center("origin")
    pymol.cmd.png(file, ray=1, dpi=600)
    pymol.cmd.zoom("origin", 1)
    pymol.cmd.delete("*")
    return file


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input PDB file")
    parser.add_argument("-o", "--output", required=True, help="Output PNG file")
    return parser.parse_args()


def main():
    args = parse_arguments()
    pdb2png(args.input, args.output)


if __name__ == "__main__":
    main()
