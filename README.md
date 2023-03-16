# VGL OZU - Night Photography Rendering Challenge @ NTIRE 2023, CVPR Workshops

Please put the test data into folder ```data/test``` before building the container.
Required weights will be automatically downloaded if not available in particular folder.

To build the docker container:

```
docker build . -t night-photo-rendering-vgl-ozu-23
```


To run the docker with GPUs:

```
docker run --rm -i --runtime=nvidia --gpus all -t night-photo-rendering-vgl-ozu-23
```

As the entrypoint, you may run the process as follows:

```
./run.sh
```

To cite the challenge report:

```
TBD
```