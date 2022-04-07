FROM ghcr.io/ml4gw/pinto:main

ENV DEEPCLEAN=/opt/deepclean
ADD . $DEEPCLEAN
WORKDIR $DEEPCLEAN/projects/microservice/cleaner

RUN pinto build .
