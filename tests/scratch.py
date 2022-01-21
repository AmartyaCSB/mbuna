from fire import Fire
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('arg1')
args, kwargs = parser.parse_known_args()

def hello(name='world'):
    print(f'hello {name}')

if __name__ == '__main__':
    Fire(hello, kwargs)
