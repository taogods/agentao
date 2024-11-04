FROM ubuntu:latest AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="$CARGO_HOME/bin:$PATH"

# Update system and install necessary packages
RUN apt-get -y update && apt-get install -y \
    make build-essential git clang curl libssl-dev llvm libudev-dev protobuf-compiler jq software-properties-common bash

# Add the Python PPA and install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get -y update && \
    apt-get install -y python3.11 python3.11-venv

# Install Rust and Cargo using bash
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
# Use bash explicitly to source the environment and update Rust
RUN /bin/bash -c "source $CARGO_HOME/env && rustup update && rustup default stable"

# Clone the Subtensor repository and set up Rust
RUN git clone https://github.com/opentensor/subtensor.git
WORKDIR /subtensor
RUN ./scripts/init.sh

# Build the binary with the faucet feature enabled
RUN /bin/bash -c "source $CARGO_HOME/env && cargo build -p node-subtensor --profile production --features pow-faucet"

RUN python3.11 -m pip install --upgrade pip

# Set up btcli
RUN git clone https://github.com/opentensor/bittensor
WORKDIR /bittensor
RUN python3.11 -m pip install --force-reinstall .
WORKDIR /

COPY ./scripts/setup_staging_subnet.sh /setup_staging_subnet.sh
RUN chmod +x /setup_staging_subnet.sh
RUN /setup_staging_subnet.sh

FROM base AS integration-test

# Set up the working directory
WORKDIR /app
COPY . .
WORKDIR /app/taoception
RUN python3.11 -m pip install --editable --force-reinstall .

CMD ["/bin/bash", "scripts/run_integration_tests.sh"]
