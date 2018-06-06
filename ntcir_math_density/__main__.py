"""
This is the command-line interface for the NTCIR Math density estimator package.
"""

from argparse import ArgumentParser
import gzip
import logging
from logging import getLogger
from pathlib import Path
import pickle
from sys import stdout

from .estimator import get_judged_identifiers, get_all_positions, get_estimators


LOG_PATH = Path("__main__.log")
LOG_FORMAT = "%(asctime)s : %(levelname)s : %(message)s"
MIN_RELEVANT_SCORE = 2
ROOT_LOGGER = getLogger()
LOGGER = getLogger(__name__)


class LabelledPath(object):
    """This class represents a path labelled with a unique single-letter label.

    Parameters
    ----------
    label : str
        A single-letter label.
    path : Path
        The labelled-path.

    Attributes
    ----------
    labels : dict of (str, Path)
        A mapping between labels, and paths.
    label : str
        A single-letter label.
    path : Path
        The labelled-path.
    """
    labels = dict()

    def __init__(self, label, path):
        assert isinstance(label, str) and len(label) == 1
        assert label not in LabelledPath.labels
        assert isinstance(path, Path)
        self.label = label
        self.path = path
        LabelledPath.labels[self.label] = self.path


def main():
    """ Main entry point of the app """
    ROOT_LOGGER.setLevel(logging.DEBUG)

    file_handler = logging.StreamHandler(LOG_PATH.open("wt"))
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    ROOT_LOGGER.addHandler(file_handler)

    terminal_handler = logging.StreamHandler(stdout)
    terminal_handler.setFormatter(formatter)
    terminal_handler.setLevel(logging.INFO)
    ROOT_LOGGER.addHandler(terminal_handler)

    LOGGER.debug("Parsing command-line arguments")
    parser = ArgumentParser(
        description="""
            Use NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR datasets to compute density, and
            probability estimators.
        """)
    parser.add_argument(
        "--datasets", nargs='+', required=True,
        type=lambda s: LabelledPath(s.split('=', 1)[0], Path(s.split('=', 1)[1])), help="""
            Paths to the directories containing the datasets. Each path must be prefixed with a
            unique single-letter label (e.g. "A=/some/path"). Note that all the datasets must be in
            the NTCIR-11 Math-2, and NTCIR-12 MathIR format, even the NTCIR-10 Math dataset.
        """)
    parser.add_argument(
        "--judgements", nargs='+', required=True,
        type=lambda s: (LabelledPath.labels[s.split(':', 1)[0]], Path(s.split(':', 1)[1])), help="""
            Paths to the files containing relevance judgements. Each path must be prefixed with
            single-letter labels corresponding to the judged datasets (e.g.
            "A:/some/path/judgement.dat"). Note that all the judgements must be in the NTCIR-11
            Math-2, and NTCIR-12 MathIR format, even the NTCIR-10 Math dataset judgements.
        """)
    parser.add_argument(
        "--output-file", type=Path, default=Path("estimators.pkl.gz"), help="""
            The path to the file, where the probability estimators will be pickled, and gzipped.
            Defaults to %(default)s.
        """)
    parser.add_argument(
        "--num-workers", type=int, default=1, help="""
            The number of processes that will be used for processing the NTCIR-10 Math dataset.
            Defaults to %(default)d.
        """)
    args = parser.parse_args()

    LOGGER.debug("Performing sanity checks on the command-line arguments")
    for dataset in args.datasets:
        assert dataset.path.exists() and dataset.path.is_dir(), dataset.path
    for _, judgement_path in args.judgements:
        assert judgement_path.exists() and judgement_path.is_file(), judgement_path
    assert args.output_file.parents[0].exists() and args.output_file.parents[0].is_dir(), \
        args.output_file
    assert args.num_workers > 0

    if args.output_file.exists():
        assert args.output_file.is_file(), args.output_file
        LOGGER.warning("%s exists", args.output_file.name)

    identifiers_judged = {}
    identifiers_relevant = {}
    for dataset_path, judgement_path in args.judgements:
        LOGGER.info(
            "Retrieving judged paragraph identifiers, and scores from %s", judgement_path.name)
        if dataset_path not in identifiers_judged:
            identifiers_judged[dataset_path] = set()
        if dataset_path not in identifiers_relevant:
            identifiers_relevant[dataset_path] = set()
        with judgement_path.open("rt") as f:
            for identifier, score in get_judged_identifiers(f):
                identifiers_judged[dataset_path].add(identifier)
                if score >= MIN_RELEVANT_SCORE:
                    identifiers_relevant[dataset_path].add(identifier)

    identifiers_all = {}
    positions_all = {}
    positions_relevant = {}
    for dataset in args.datasets:
        LOGGER.info(
            "Retrieving all paragraph identifiers, and positions from %s", dataset.path.name)
        identifiers_all[dataset.path] = []
        positions_all[dataset.path] = []
        positions_relevant[dataset.path] = []
        for directory, identifier, position in get_all_positions(dataset.path, args.num_workers):
            identifiers_all[dataset.path].append((directory, identifier))
            positions_all[dataset.path].append(position)
            if identifier in identifiers_relevant[dataset.path]:
                positions_relevant[dataset.path].append(position)

    for dataset in args.datasets:
        LOGGER.info(
            "%d / %d / %d relevant / judged / total identifiers in dataset %s",
            len(identifiers_relevant[dataset.path]), len(identifiers_judged[dataset.path]),
            len(identifiers_all[dataset.path]), dataset.path.name)

    LOGGER.info("Fitting density, and probability estimators")
    estimators = get_estimators(positions_all, positions_relevant)

    LOGGER.info("Pickling %s", args.output_file.name)
    with gzip.open(args.output_file.open("wb"), "wb") as f:
        pickle.dump(estimators, f)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
