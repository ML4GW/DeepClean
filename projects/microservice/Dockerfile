FROM ghcr.io/ml4gw/pinto:latest
ENV DEEPCLEAN=/opt/deepclean
COPY . $DEEPCLEAN

ARG project
ENV FLASK_APP=$DEEPCLEAN/projects/microservice/$project/$project
RUN pinto -p $DEEPCLEAN/projects/microservice/$project build
