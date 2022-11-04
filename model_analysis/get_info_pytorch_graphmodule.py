import argparse

import torch
from torchsummary import summary

from thop import profile

G = 10**9
M = 10**6


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="get_info_graphmodule", add_help=add_help)
    parser.add_argument('--model')
    return parser



if __name__ == "__main__":
    args=get_args_parser().parse_args()
    device = 'cpu'
    dummy_input = torch.ones((1,3,520,520)).to(device)
    
    model = torch.load(args.model).to(device)
    macs, params = profile(model, inputs=(dummy_input, ))
    result_txt = f" Params: {params/M:.2f}M\nFLOPs: {(macs*2)/G:.2f}G\n"
    print(args.model)
    print(result_txt)

    with open('/root/workspace/info_result.txt', 'a') as f:
        f.write(f'{args.model}\n')
        f.write(f'{result_txt}\n')
