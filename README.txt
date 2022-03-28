# Data preparation

## Docker command for running 'rec_view_extractor':

```
xhost local:docker
nvidia-docker run -it -v /home/jbweibel/code/gronet:/home/code -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY gronet
```

## Command to run reconstruction:

```
~/code/mapping/mapping/mapping -i /home/jbweibel/code/gronet/docker_build/bottle/track_0/ --scaleDepth 0.1 --scaleGroundtruth 0.5 --headless --detailed /home/jbweibel/code/gronet/bottle/model_0[_gt].ply [-t]
```

## Update the model to have vertex_indices (PCL reader compatible)

```
sed 's/vertex_index/vertex_indices/' model_2_gt.ply > model_2_gt_new.ply
```


# Experiments

## Building gronet-tf:

```
cd images
docker build -t gronet-tf -f Dockerfile.tf --network my-net .
```

## Running gronet-tf

```
nvidia-docker run --network my-net -it  -v /home/jbweibel/:/home/jbweibel -p 6006:6006 -p 8080:8080 gronet-tf
nvidia-docker run --network my-net -it  -v /home/jbweibel/:/home/jbweibel -v /mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset/:/mnt/a85a47fd-9581-4557-8463-5459be4df92b/dataset -p 6006:6006 -p 8080:8080 gronet-tf  # With the dir for ScanNet
docker exec -it <TAB> bash # There's probably only one container started
```

It will start a jupyter notebook in the main instance, hence docker exec ... cool ...

# Others

/!\ REMOVED FILE:

/home/jbweibel/dataset/ModelNet/ModelNet10_TrainPly/bathtub/bathtub_0023_bin.ply
