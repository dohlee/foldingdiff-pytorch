import argparse
import imageio.v2 as imageio
import glob

from pygifsicle import optimize


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Input directory containing png files"
    )
    parser.add_argument("-o", "--output", required=True, help="Output PNG file")
    return parser.parse_args()


def main():
    args = parse_arguments()

    filenames = glob.glob(f"{args.input}/*.png")

    with imageio.get_writer(args.output, mode="I", loop=0, duration=0.01) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

            if filename.endswith("_1000.png"):
                for _ in range(500):
                    writer.append_data(image)


if __name__ == "__main__":
    main()
