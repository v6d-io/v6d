FROM python:3.10

RUN pip3 install --no-cache-dir pandas requests scikit-learn numpy vineyard

WORKDIR /

ARG APP
ENV APP ${APP}

COPY ${APP} /${APP}
