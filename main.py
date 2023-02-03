#!/usr/bin/env python3
"""Entry point to training/validating/testing for a user given experiment
name."""
import allenact.main

import sys
import os
sys.path.append(os.path.join(os.path.abspath('.'), 'projects/ithor_A2SP'))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    allenact.main.main()
