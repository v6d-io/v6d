ARG PLATFORM=x86_64
FROM ghcr.io/v6d-io/v6d/vineyardd:alpine-latest_x86_64 as base_x86_64
FROM ghcr.io/v6d-io/v6d/vineyardd:alpine-latest_aarch64  as base_aarch64

FROM base_$PLATFORM

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
