# STSAGNN
Code implementation of Spatio-Temporal Synchronous Adaptive Graph Neural Network

The dataset is available on [Google Drive](https://drive.google.com/file/d/1RYR2q_aXzhhe4eOMioBxYMrSeIDbY-Pw/view?usp=drive_link),
to use, download then unzip to the root menu.

We also provide separate data of [NYCTaxi](https://drive.google.com/file/d/1q3ZbuEJALkEsSXVRKuaCatW6_MavKJpM/view?usp=drive_link)
and [NYCBike](https://drive.google.com/file/d/1W4gKCeFqUvvZ8JLjVNfPYyQCjBKmERWd/view?usp=drive_link).
The data is a .npy file with shape of 17520×66×66 and 17520×75×75, respectively.

To run the model, you can simply
```bash
python main.py
```
This will run STSAGNN on NYCTaxi dataset.

If you want to run another model of dataset, you can
```bash
python main.py --model STSAGNN --dataset NYC_bike
```

If you want to run multiple dataset in the same time, you can
```bash
python main.py --model STSAGNN --data_list ['NYC_2023', 'NYC_bike', 'HZ_Metro', 'SH_Metro'] --debug_mode 2
```
This is all the datasets currently available.

We currently support the following running modes:

- 0: run ONE model on ONE dataset
- 1: run MULTIPLE model on ONE dataset
- 2: run ONE model on MULTIPLE dataset
- 3: run ONE model on ONE dataset, with MULTIPLE parameter setting
- 4: run MULTIPLE model on MULTIPLE dataset

Besides main model, I also provide a transformer-variate which replace GRU with transformer, named STSformer,
you can run
```bash
python main.py --model STSformer --dataset NYC_bike
```
Sadly the transformer version perform worse than GRU version
