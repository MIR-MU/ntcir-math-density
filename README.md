# Introduction
NTCIR Math Density Estimator is a Python 3 command-line utility that computes
and plots density, and probability estimates from judged datasets in the
[NTCIR-11 Math-2][paper:aizawaetal14-ntcir11], and [NTCIR-12
MathIR][paper:zanibbi16-ntcir12] format. Most importantly, the package
estimates the probability `P(relevant | position)`, where `position` is a
position of a paragraph in a document.

[paper:aizawaetal13-ntcir10]: https://ntcir-math.nii.ac.jp/wp-content/blogs.dir/23/files/2013/10/01-NTCIR10-OV-MATH-AizawaA.pdf (NTCIR-10 Math Pilot Task Overview, Proceedings of the 10th NTCIR Conference, June 18–21, 2013, Tokyo, Japan)
[paper:aizawaetal14-ntcir11]: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.686.444&rep=rep1&type=pdf (NTCIR-11 Math-2 Task Overview, Proceedings of the 11th NTCIR Conference, December 9–12, 2014, Tokyo, Japan)
[paper:zanibbi16-ntcir12]: https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings12/pdf/ntcir/OVERVIEW/01-NTCIR12-OV-MathIR-ZanibbiR.pdf (NTCIR-12 MathIR Task Overview, Proceedings of the 12th NTCIR Conference on Evaluation of Information Access Technologies, June 7–10, 2016 Tokyo Japan)

[www:ntcir-task-data]: https://www.nii.ac.jp/dsc/idr/en/ntcir/ntcir-taskdata.html (Downloading NTCIR Test Collections Task Data)
[www:ntcir-10-math-data]: https://ntcir-math.nii.ac.jp/data/ (NTCIR-12 MathIR » Data » NTCIR-10 Math Pilot Task)
[www:ntcir-12-mathir-data]: https://ntcir-math.nii.ac.jp/data/ (NTCIR-12 MathIR » Data » NTCIR-12 MathIR Pilot Task)

# Usage
Installing:

    $ pip install ntcir-math-density

Displaying the usage:

    $ ntcir-math-density --help
    usage: ntcir-math-density [-h] [--datasets DATASETS [DATASETS ...]]
                              [--judgements JUDGEMENTS [JUDGEMENTS ...]]
                              [--plots PLOTS [PLOTS ...]] [--positions POSITIONS]
                              [--estimates ESTIMATES] [--num-workers NUM_WORKERS]

    Use NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR datasets to compute
    density, and probability estimates.

    optional arguments:
    -h, --help            show this help message and exit
    --datasets DATASETS [DATASETS ...]
                            Paths to the directories containing the datasets. Each
                            path must be prefixed with a unique single-letter
                            label (e.g. "A=/some/path"). Note that all the
                            datasets must be in the NTCIR-11 Math-2, and NTCIR-12
                            MathIR format, even the NTCIR-10 Math dataset.
    --judgements JUDGEMENTS [JUDGEMENTS ...]
                            Paths to the files containing relevance judgements.
                            Each path must be prefixed with single-letter labels
                            corresponding to the judged datasets (e.g.
                            "A:/some/path/judgement.dat"). Note that all the
                            judgements must be in the NTCIR-11 Math-2, and
                            NTCIR-12 MathIR format, even the NTCIR-10 Math dataset
                            judgements.
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
                            processing the NTCIR-10 Math dataset, and for
                            computing the density, and probability estimates.
                            Defaults to 1.

Extracting density, and probability estimates, and plotting the estimates using 64 worker processes:

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

Extracting density, and probability estimates using 64 worker processes:

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

Plotting the estimates using 64 worker processes:

    $ ntcir-math-density --num-workers 64 \
    >     --estimates estimates.pkl.gz --plots plot.pdf plot.svg
    Unpickling estimates.pkl.gz
    Plotting plot.svg
    Plotting plot.pdf
