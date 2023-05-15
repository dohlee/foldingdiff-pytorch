import argparse
import torch
import math

from tqdm import tqdm

from foldingdiff_pytorch import FoldingDiff
from foldingdiff_pytorch.util import wrap

DEFAULT_MU = [
    -1.311676263809204,
    0.620250940322876,
    0.3829933702945709,
    1.940455198287964,
    2.0217323303222656,
    2.108278274536133
]

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='Model checkpoint')
    parser.add_argument('--timepoints', type=int, default=1000)
    parser.add_argument('--num-residues', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--mu', type=int, nargs='+', default=DEFAULT_MU)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_argument()
    T = args.timepoints
    mu = torch.tensor(args.mu).float()

    # Load model
    model = FoldingDiff()

    state_dict = torch.load(args.ckpt)['state_dict']
    model.load_state_dict(state_dict)

    model.eval()

    s = 8e-3
    t = torch.arange(T + 1)
    f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
    alpha_bar = f_t / f_t[0]
    beta = torch.cat([torch.tensor([0.0]), torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], min=1e-5, max=1 - 1e-5)])
    alpha = 1 - beta

    trajectory = []
    with torch.no_grad():

        x = wrap(torch.randn(args.batch_size, args.num_residues, 6))
        trajectory.append(x.unsqueeze(1))

        for t in tqdm(range(T, 0, -1), desc='sampling'):
            sigma_t = math.sqrt( (1 - alpha_bar[t-1]) / (1 - alpha_bar[t]) * beta[t] )

            # Sample from N(0, sigma_t^2)
            if t > 1:
                z = torch.randn(args.batch_size, args.num_residues, 6) * sigma_t
            else:
                z = torch.zeros(args.batch_size, args.num_residues, 6)

            # Update x
            t_tensor = torch.tensor([t]).long().unsqueeze(0)
            x = wrap( 1 / math.sqrt(alpha[t]) * (x - beta[t] / math.sqrt(1 - alpha_bar[t]) * model(x, t_tensor)) + z)

            trajectory.append(x.unsqueeze(1))

    trajectory = wrap( torch.cat(trajectory, dim=1) + mu )
    torch.save(trajectory, args.output)

if __name__ == '__main__':
    main()