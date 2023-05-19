# Setting Up

Run the following commands in the given order:

```
conda create -n qblox python=3.9
conda activate qblox
pip install qblox-instruments
pip install quantify-core
pip install quantify-scheduler
```


In the first two lines, qblox is the name of the enviroment, we can rename it to whatever we want.
The command: `pip show qblox-instruments` can show the version used of qblox-instruments and so on
Commands such as `conda env list` can show the environments that have been created.

For more info visit this [site](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/getting_started/setup.html)
Once the packages are installed and we want to run the python scripts, it must run from that environment: 
Setup PyCharm to use a [Conda environment](https://www.youtube.com/watch?v=n1SFlh-pW_Q&ab_channel=DennisMadsen)

