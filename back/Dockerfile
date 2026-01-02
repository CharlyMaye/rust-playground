FROM ghcr.io/prulloac/base:bookworm

# Install Rust with default stable toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default && \
    . "$HOME/.cargo/env" && \
    rustup default stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install additional tools
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/back

# Keep container running
CMD ["sleep", "infinity"]
