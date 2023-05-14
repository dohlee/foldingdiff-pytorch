import argparse
import torch
import math

from tqdm import tqdm

from foldingdiff_pytorch import FoldingDiff
from foldingdiff_pytorch.util import wrap


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='Model checkpoint')
    parser.add_argument('--timepoints', type=int, default=1000)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_argument()
    T = args.timepoints

    # Load model
    model = FoldingDiff()

    state_dict = torch.load(args.ckpt)['state_dict']
    model.load_state_dict(state_dict)

    model.eval()

    s = 8e-3
    t = torch.arange(T + 1)
    f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2.0).square()
    alpha_bar = f_t / f_t[0]
    beta = torch.cat([torch.tensor([0.0]), torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], min=0.001, max=0.999)])
    alpha = 1 - beta

    trajectory = []
    with torch.no_grad():

        x = wrap(torch.randn(1, 64, 6))
        trajectory.append(x)

        for t in tqdm(range(T, 0, -1), desc='sampling'):
            sigma_t = math.sqrt( (1 - alpha_bar[t-1]) / (1 - alpha_bar[t]) * beta[t] )

            # Sample from N(0, sigma_t^2)
            if t > 1:
                z = torch.randn(1, 64, 6) * sigma_t
            else:
                z = torch.zeros(1, 64, 6)

            # Update x
            t_tensor = torch.tensor([t]).long().unsqueeze(0)
            x = wrap( 1 / math.sqrt(alpha[t]) * (x - beta[t] / math.sqrt(1 - alpha_bar[t]) * model(x, t_tensor)) + z)

            trajectory.append(x)

    trajectory = torch.cat(trajectory, dim=0)
    torch.save(trajectory, args.output)

if __name__ == '__main__':
    main()