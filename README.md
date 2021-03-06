NTCIR Math Density Estimator – Estimates relevance of documents based on data from NTCIR Math tasks
===================================================================================================
[![CircleCI](https://circleci.com/gh/MIR-MU/ntcir-math-density/tree/master.svg?style=shield)][ci]

 [ci]: https://circleci.com/gh/MIR-MU/ntcir-math-density/tree/master (CircleCI)

NTCIR Math Density Estimator is a Python 3 command-line utility that uses
datasets, and judgements in the [NTCIR-11 Math-2][aizawaetal14-ntcir11], and
[NTCIR-12 MathIR][zanibbi16-ntcir12] XHTML5 format to compute density, and
probability estimates. Most importantly, the package estimates the probability
`P(relevant | position)`, where `position` is a position of a paragraph in a
document.

[aizawaetal14-ntcir11]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.686.444&rep=rep1&type=pdf (NTCIR-11 Math-2 Task Overview)
[zanibbi16-ntcir12]: https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings12/pdf/ntcir/OVERVIEW/01-NTCIR12-OV-MathIR-ZanibbiR.pdf (NTCIR-12 MathIR Task Overview)

Usage
=====
Installing
----------
The package can be installed by executing the following command:

    $ pip install ntcir-math-density

Displaying the usage
--------------------
Usage information for the package can be displayed by executing the following
command:

    $ ntcir-math-density --help
    usage: ntcir-math-density [-h] [--datasets DATASETS [DATASETS ...]]
                              [--judgements JUDGEMENTS [JUDGEMENTS ...]]
                              [--plots PLOTS [PLOTS ...]] [--positions POSITIONS]
                              [--estimates ESTIMATES] [--num-workers NUM_WORKERS]

    Use datasets, and judgements in NTCIR-11 Math-2, and NTCIR-12 MathIR XHTML5
    format to compute density, and probability estimates.

    optional arguments:
    -h, --help            show this help message and exit
    --datasets DATASETS [DATASETS ...]
                            Paths to the directories containing the datasets. Each
                            path must be prefixed with a unique single-letter
                            label followed by an equals sign (e.g. "A=/some/path").
    --judgements JUDGEMENTS [JUDGEMENTS ...]
                            Paths to the files containing relevance judgements.
                            Each path must be prefixed with a single-letter label
                            corresponding to the judged dataset followed by a
                            semicolon (e.g. "A:/some/path/judgement.dat").
    --plots PLOTS [PLOTS ...]
                            The path to the files, where the probability
                            estimates will plotted. When no datasets are
                            specified, the estimates file will be loaded.
    --positions POSITIONS
                            The path to the file, where the estimated positions of
                            all paragraph identifiers from all datasets will be
                            stored. Defaults to positions.pkl.gz.
    --estimates ESTIMATES
                            The path to the file, where the density, and
                            probability estimates will be stored. When no
                            datasets are specified, this file will be loaded to
                            provide the estimates for plotting. Defaults to
                            estimates.pkl.gz.
    --num-workers NUM_WORKERS
                            The number of processes that will be used for
                            processing the datasets, and for computing the
                            density, and probability estimates. Defaults to 1.

Extracting estimates
--------------------
The following command extracts density, and probability estimates and plots the
estimates using 64 worker processes:

    $ ntcir-math-density --num-workers 64 \
    >     --datasets A=ntcir-10-converted B=ntcir-11-12 \
    >     --judgements A:NTCIR_10_Math-qrels_fs-converted.dat A:NTCIR_10_Math-qrels_ft-converted.dat \
    >                  B:NTCIR11_Math-qrels.dat B:NTCIR12_Math-qrels_agg.dat \
    >                  B:NTCIR12_Math_simto-qrels_agg.dat \
    >     --estimates estimates.pkl.gz --positions positions.pkl.gz \
    >     --plots plot.pdf plot.svg
    Retrieving judged paragraph identifiers, and scores from NTCIR_10_Math-qrels_fs-converted.dat
    100%|█████████████████████████████████████████████████████| 2129/2129 [00:00<00:00, 334959.05it/s]
    Retrieving judged paragraph identifiers, and scores from NTCIR_10_Math-qrels_ft-converted.dat
    100%|█████████████████████████████████████████████████████| 1425/1425 [00:00<00:00, 353201.94it/s]
    Retrieving judged paragraph identifiers, and scores from NTCIR11_Math-qrels.dat
    100%|█████████████████████████████████████████████████████| 2500/2500 [00:00<00:00, 343345.12it/s]
    Retrieving judged paragraph identifiers, and scores from NTCIR12_Math-qrels_agg.dat
    100%|█████████████████████████████████████████████████████| 4251/4251 [00:00<00:00, 342252.50it/s]
    Retrieving judged paragraph identifiers, and scores from NTCIR12_Math_simto-qrels_agg.dat
    100%|█████████████████████████████████████████████████████| 654/654 [00:00<00:00, 314428.57it/s]
    Retrieving all paragraph identifiers, and positions from ntcir-10-converted
    get_all_identifiers(ntcir-10-converted): 5405167it [04:30, 19946.57it/s]
    get_all_positions(ntcir-10-converted): 100%|█████████| 5405167/5405167 [08:44<00:00, 10306.72it/s]
    Retrieving all paragraph identifiers, and positions from ntcir-11-12
    get_all_identifiers(ntcir-11-12): 8301578it [08:08, 16985.19it/s]
    get_all_positions(ntcir-11-12): 100%|█████████████████| 8301578/8301578 [44:30<00:00, 3108.88it/s]
    1043 / 3146 / 5405167 relevant / judged / total identifiers in dataset ntcir-10-converted
    1742 / 7059 / 8301578 relevant / judged / total identifiers in dataset ntcir-11-12
    Pickling positions.pkl.gz
    Fitting density, and probability estimators
    Fitting prior p(position) density estimator
    Fitting conditional p(position | relevant) density estimator
    Computing density, and probability estimates
    p(position): 100%|████████████████████████████████████████████████| 64/64 [01:19<00:00,  1.24s/it]
    p(position | relevant): 100%|█████████████████████████████████████| 64/64 [01:19<00:00,  1.24s/it]
    Pickling estimates.pkl.gz
    Plotting plot.svg
    Plotting plot.pdf

The following command extracts density, and probability estimates using 64 worker processes:

    $ ntcir-math-density --num-workers 64 \
    >     --datasets A=ntcir-10-converted B=ntcir-11-12 \
    >     --judgements A:NTCIR_10_Math-qrels_fs-converted.dat A:NTCIR_10_Math-qrels_ft-converted.dat \
    >                  B:NTCIR11_Math-qrels.dat B:NTCIR12_Math-qrels_agg.dat \
    >                  B:NTCIR12_Math_simto-qrels_agg.dat \
    >     --estimates estimates.pkl.gz --positions positions.pkl.gz
    Retrieving judged paragraph identifiers, and scores from NTCIR_10_Math-qrels_fs-converted.dat
    100%|█████████████████████████████████████████████████████| 2129/2129 [00:00<00:00, 334959.05it/s]
    Retrieving judged paragraph identifiers, and scores from NTCIR_10_Math-qrels_ft-converted.dat
    100%|█████████████████████████████████████████████████████| 1425/1425 [00:00<00:00, 353201.94it/s]
    Retrieving judged paragraph identifiers, and scores from NTCIR11_Math-qrels.dat
    100%|█████████████████████████████████████████████████████| 2500/2500 [00:00<00:00, 343345.12it/s]
    Retrieving judged paragraph identifiers, and scores from NTCIR12_Math-qrels_agg.dat
    100%|█████████████████████████████████████████████████████| 4251/4251 [00:00<00:00, 342252.50it/s]
    Retrieving judged paragraph identifiers, and scores from NTCIR12_Math_simto-qrels_agg.dat
    100%|█████████████████████████████████████████████████████| 654/654 [00:00<00:00, 314428.57it/s]
    Retrieving all paragraph identifiers, and positions from ntcir-10-converted
    get_all_identifiers(ntcir-10-converted): 5405167it [04:30, 19946.57it/s]
    get_all_positions(ntcir-10-converted): 100%|█████████| 5405167/5405167 [08:44<00:00, 10306.72it/s]
    Retrieving all paragraph identifiers, and positions from ntcir-11-12
    get_all_identifiers(ntcir-11-12): 8301578it [08:08, 16985.19it/s]
    get_all_positions(ntcir-11-12): 100%|█████████████████| 8301578/8301578 [44:30<00:00, 3108.88it/s]
    1043 / 3146 / 5405167 relevant / judged / total identifiers in dataset ntcir-10-converted
    1742 / 7059 / 8301578 relevant / judged / total identifiers in dataset ntcir-11-12
    Pickling positions.pkl.gz
    Fitting density, and probability estimators
    Fitting prior p(position) density estimator
    Fitting conditional p(position | relevant) density estimator
    Computing density, and probability estimates
    p(position): 100%|████████████████████████████████████████████████| 64/64 [01:19<00:00,  1.24s/it]
    p(position | relevant): 100%|█████████████████████████████████████| 64/64 [01:19<00:00,  1.24s/it]
    Pickling estimates.pkl.gz

The following command plots the estimates using 64 worker processes:

    $ ntcir-math-density --num-workers 64 \
    >     --estimates estimates.pkl.gz --plots plot.pdf plot.svg
    Unpickling estimates.pkl.gz
    Plotting plot.svg
    Plotting plot.pdf
