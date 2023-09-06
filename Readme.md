# [Predicting COVID-19 positivity and hospitalization with multi-scale graph neural networks](https://www.nature.com/articles/s41598-023-31222-6)


## Data


### Labels

We gather the ground truth for number of confirmed cases per region through open data for [France](https://www.data.gouv.fr/en/datasets/donnees-relatives-aux-tests-de-depistage-de-covid-19-realises-en-laboratoire-de-ville/).
We have preprocessed the data and the final versions are located in the data folder.


### Graphs

The graphs are formed using the movement data from facebook Data For Good disease prevention [maps](https://dataforgood.fb.com/docs/covid19/). More specifically, we used the total number of people moving daily from one region to another, using the [Movement between Administrative Regions](https://dataforgood.fb.com/tools/movement-range-maps/) datasets. We can not share the initial data due to the data license agreement, but after contacting the FB Data for Good team, we reached the consensus that we can share an aggregated and diminished version which was used for our experiments.
These can be found inside the "graphs" folder of each country. These include the mobility maps between administrative regions that we use in our experiments starting from 27/12/2020 for France until 27/06/2021.


## Code

### Requirements
To run this code you will need the following python and R packages:
[numpy](https://www.numpy.org/), [pandas](https://pandas.pydata.org/), [scipy](https://www.scipy.org/) ,[pytorch 1.5.1](https://pytorch.org/), [pytorch-geometric 1.5.0](https://github.com/rusty1s/pytorch_geometric), [networkx 1.11](https://networkx.github.io/), [sklearn](https://scikit-learn.org/stable/), dplyr, sf, ggplot2, sp.

#### Requirements for MAC
For MAC users, please use these versions: torch 1.7.0, torch-cluster 1.5.9 , torch-geometric 2.0.1 , torch-scatter 2.0.7, torch-sparse 0.6.12, torch-spline-conv 1.2.1., pystan 2.18.0.0 (for FB prophet).


### Run
To run the experiments with the default settings:

```bash

cd code

python experiments_regression.py

```

## Citation

If you find the methods or the datasets useful in your research, please consider adding the following citation:

```bibtex
@article{skianis2023predicting,
  title={Predicting COVID-19 positivity and hospitalization with multi-scale graph neural networks},
  author={Skianis, Konstantinos and Nikolentzos, Giannis and Gallix, Benoit and Thiebaut, Rodolphe and Exarchakis, Georgios},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={5235},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
**License**

- [MIT License](https://github.com/geopanag/pandemic_tgnn/blob/master/LICENSE)
