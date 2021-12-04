from crocodile.generator import TrainParams, load_generator
from crocodile.executor import load_executor, ExecutorConfig
from simple_parsing import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExecutorConfig, dest="executor")
    parser.add_arguments(TrainParams, dest="train")
    args = parser.parse_args()

    executor = load_executor(ExecutorConfig)
    generator = load_generator(args.train.generator)
    executor(generator.train, args.train)