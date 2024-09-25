# Probabilistically Enriched Transformers Applied on Neural Spatio-temporal Point Processes

## Papers

Erfanian, Negar, Santiago Segarra, and Maarten V. de Hoop. "Neural multi-event forecasting on spatio-temporal point processes using probabilistically enriched transformers." (2022).

### 1. Survey papers



## Configure docker container on the server

First clone project in your desired path:

```bash
git clone git@github.com:Negar-Erfanian/Neural-spatio-temporal-probabilistic-transformers.git
```

In ssh command, first pull docker image

```bash
docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter
```

Then launch the container

```bash
docker run --user $(id -u):$(id -g) --runtime=nvidia --rm -it -v ~/Neural-spatio-temporal-probabilistic-transformers:/tensorflow/Neural-spatio-temporal-probabilistic-transformers -w /tensorflow/Neural-spatio-temporal-probabilistic-transformers -p 8300:8888 tensorflow/tensorflow:latest-gpu-py3-jupyter
```

Access the jupyter interface of the container from browser, and launch a terminal from jupyter

```bash
pip3 install -r requirements.txt
```

Test if everything is ok from the jupyter terminal

```bash
python3 train.py
```


Install any missing packages in the error message

If test pass, commit the container from ssh command window

Check container id `docker container ls`, assuming it is `c3f279d17e0a`

Then commit the changes to the image of the same name

```bash
docker commit c3f279d17e0a tensorflow/tensorflow:latest-gpu-py3-jupyter
```


The newly installed python packages are now ready for next time you launch the container.
