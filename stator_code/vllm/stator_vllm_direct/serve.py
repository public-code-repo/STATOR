import sys

import stator_vllm
import stator_vllm_direct

stator_vllm.accept_device_names()
stator_vllm_direct.register()


def main():
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    return vllm_main()


if __name__ == "__main__":
    sys.exit(main())
