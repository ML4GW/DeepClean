FROM ghcr.io/ml4gw/pinto:latest
ENV DEEPCLEAN=/opt/deepclean
COPY . $DEEPCLEAN

ARG project
ENV PROJECT_DIR=$DEEPCLEAN/projects/microservice/$project
RUN pinto -p $PROJECT_DIR build
