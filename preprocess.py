import dataclasses
import datetime

import click
import os
import pathlib
import shutil

import pandas as pd


@dataclasses.dataclass
class InputOutputDir:
    input: str
    output: str


@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', default='output')
@click.option('-z', '--zfill', type=int, default=0)
def main(input, output, zfill):
    dt_now = datetime.datetime.now()
    dt_now = dt_now.strftime('%d-%m-%Y-%H-%M-%S')

    if not os.path.exists(input):
        raise FileNotFoundError(input)

    output = os.path.join(output, dt_now)
    os.makedirs(output, exist_ok=True)

    out_dirs = list()
    for i, p in enumerate(sorted(pathlib.Path(input).glob('*')), start=1):
        if p.is_dir():
            idx = str(i).zfill(zfill)
            dst = os.path.join(output, 'f' + idx)
            shutil.copytree(input, dst)
            out_dirs.append(InputOutputDir(input, dst))

    df = pd.DataFrame(data=out_dirs, columns=['input', 'output'])
    filename = os.path.join(output, 'input.csv')
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding='utf-8', index=False, header=True)


if __name__ == '__main__':
    main()