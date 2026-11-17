import sys

import stator_vllm

stator_vllm.accept_device_names()


def main():
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    return vllm_main()


if __name__ == "__main__":
    sys.exit(main())
