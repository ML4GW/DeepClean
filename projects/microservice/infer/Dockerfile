FROM ghcr.io/ml4gw/pinto:latest

ENV DEEPCLEAN=/opt/deepclean
ADD . $DEEPCLEAN
WORKDIR $DEEPCLEAN/projects/microservice/infer

RUN pinto build
