### SPECFM2D setup
- Pull the docker image - `docker pull ghcr.io/seisscoped/adjtomo:latest`
- Launch the container JupyterHub - 
    ```bash
    docker run -p 8888:8888 \
        --mount type=bind,source=$(pwd),target=/home/scoped/work \
        ghcr.io/seisscoped/adjtomo:latest
    ```
- Run the example with `seisflows examples run 3 -r /home/scoped/specfem2d`