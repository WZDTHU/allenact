import os
from pathlib import Path

from setuptools import find_packages, setup


def parse_req_file(fname, initial=None):
    """Reads requires.txt file generated by setuptools and outputs a
    new/updated dict of extras as keys and corresponding lists of dependencies
    as values.

    The input file's contents are similar to a `ConfigParser` file, e.g.
    pkg_1
    pkg_2
    pkg_3

    [extras1]
    pkg_4
    pkg_5

    [extras2]
    pkg_6
    pkg_7
    """
    reqs = {} if initial is None else initial
    cline = None
    with open(fname, "r") as f:
        for line in f.readlines():
            line = line[:-1].strip()
            if len(line) == 0:
                continue
            if line[0] == "[":
                # Add new key for current extras (if missing in dict)
                cline = line[1:-1].strip()
                if cline not in reqs:
                    reqs[cline] = []
            else:
                # Only keep dependencies from extras
                if cline is not None:
                    reqs[cline].append(line)
    return reqs


def get_version(fname):
    """Reads PKG-INFO file generated by setuptools and extracts the Version
    number."""
    res = "UNK"
    with open(fname, "r") as f:
        for line in f.readlines():
            line = line[:-1]
            if line.startswith("Version:"):
                res = line.replace("Version:", "").strip()
                break
    if res in ["UNK", ""]:
        raise ValueError(f"Missing Version number in {fname}")
    return res


def _do_setup():
    base_dir = os.path.abspath(os.path.dirname(Path(__file__)))

    if not os.path.exists(
        os.path.join(base_dir, "allenact.egg-info/dependency_links.txt")
    ):
        # Build mode for sdist
        os.chdir(os.path.join(base_dir, ".."))

        with open(".VERSION", "r") as f:
            __version__ = f.readline().strip()

        # Extra dependencies for development (actually unnecessary)
        extras = {
            "dev": [
                l.strip()
                for l in open("dev_requirements.txt", "r").readlines()
                if l.strip() != ""
            ]
        }
    else:
        # Install mode from sdist
        __version__ = get_version(os.path.join(base_dir, "allenact.egg-info/PKG-INFO"))
        extras = parse_req_file(
            os.path.join(base_dir, "allenact.egg-info/requires.txt")
        )

    setup(
        name="allenact",
        version=__version__,
        description="AllenAct framework",
        long_description=(
            "AllenAct is a modular and flexible learning framework designed with"
            " a focus on the unique requirements of Embodied-AI research."
        ),
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        keywords=["reinforcement learning", "embodied-AI", "AI", "RL", "SLAM"],
        url="https://github.com/allenai/allenact",
        author="Allen Institute for Artificial Intelligence",
        author_email="lucaw@allenai.org",
        license="MIT",
        packages=find_packages(include=["allenact", "allenact.*"]),
        install_requires=[
            "gym>=0.17.0,<0.20.0",
            "torch>=1.6.0,!=1.8.0,<2.0.0",
            "tensorboardx>=2.1",
            "torchvision>=0.7.0",
            "setproctitle",
            "moviepy>=1.0.3",
            "filelock",
            "numpy>=1.19.1",
            "Pillow>=8.2.0,<9.0.0",
            "matplotlib>=3.3.1",
            "networkx",
            "opencv-python",
            "wheel>=0.36.2",
            "attrs>=21.4.0",
            "scipy>=1.5.4",
        ],
        setup_requires=["pytest-runner"],
        tests_require=["pytest", "pytest-cov", "compress_pickle"],
        entry_points={"console_scripts": ["allenact=allenact.main:main"]},
        extras_require=extras,
    )


if __name__ == "__main__":
    _do_setup()
