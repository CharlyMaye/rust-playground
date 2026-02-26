# Rust Monorepo

This repository contains three Rust projects at different stages of activity.

**Current focus: [`test-neural/`](test-neural/README.md)** — a deep learning ecosystem built from scratch in Rust, targeting WebAssembly and the browser. The two other projects (`back/` and `docker-back/`) are currently **on pause**.

Everything runs inside a dev container — no local Rust or toolchain installation required.

---

## Repository Layout

```
/
├── test-neural/       # ← ACTIVE  — Deep learning ecosystem (dense nets → CNNs → WASM)
├── back/              # ← PAUSED  — Actix-web HTTP server with session authentication
├── docker-back/       # ← PAUSED  — Rust tool for programmatic Docker & Compose management
│
├── Dockerfile         # Dev container image (Rust stable + wasm-pack)
└── docker-compose.yml # Orchestrates `back` + Redis
```

---

## Active Project

### `test-neural/` — Deep Learning in Rust → WebAssembly

A self-contained deep learning ecosystem written from scratch in Rust, designed to be understood layer by layer — from the mathematics of a single neuron up to convolutional architectures running live in the browser via WebAssembly.

The goal is not to replace PyTorch. It is to *understand* what PyTorch does, by building it.

| Crate | Status | Responsibility |
|---|---|---|
| `cma-neural-network` | ✅ Active | Dense layers, 15+ activations, 5 optimizers, regularization, metrics |
| `cma-cnn` | 🚧 In progress | Conv2D, MaxPool2D, BatchNorm2D, Depthwise Conv |
| `cma-autograd` | 🚧 In progress | Dynamic computation graph, automatic gradient computation |
| `cma-models` | 🚧 In progress | LeNet-5, AlexNet, VGG-16, ResNet, EfficientNet |

Trained models are compiled to `.wasm` via `wasm-pack` and served by an Angular app or plain HTML pages — inference runs entirely in the browser with no backend.

**→ Full documentation: [`test-neural/README.md`](test-neural/README.md)**

---

## Paused Projects

These projects are functional but not currently under active development.

### `back/` — Actix-web Backend *(paused)*

A minimal but complete HTTP server built on [Actix-web 4](https://actix.rs/) with Redis-backed session management.

- Session middleware with `actix-session` + Redis store
- Two authentication patterns: per-route **Extractor** and scope-level **Middleware**
- Static file serving, structured logging

**Stack:** Actix-web · actix-session · Redis · serde

```bash
docker compose up   # server at http://localhost:8080
```

---

### `docker-back/` — Docker Management Utility *(paused)*

A Rust CLI that talks to the Docker daemon and mutates `docker-compose.yml` files programmatically, without shelling out to the `docker` CLI.

- Manage images, containers, and networks via [Bollard](https://github.com/fussybeaver/bollard)
- Parse and modify `docker-compose.yml` at runtime via `serde_yaml`

**Stack:** Bollard · Tokio · serde-yaml

```bash
cd docker-back && cargo run
```

---

## Dev Container

The `Dockerfile` sets up a complete development environment: Rust stable, `wasm32-unknown-unknown` target, `wasm-pack`, `wasm-bindgen-cli`, and Git.

```bash
# Recommended: open in VS Code Dev Container
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

---

## Requirements

| Tool | Purpose |
|---|---|
| Rust 1.70+ | All projects |
| `wasm-pack` | WASM builds in `test-neural/` |
| Docker + Compose | Running `back/` + Redis |
| Node.js 18+ | Angular app in `test-neural/ai-web-app/` |

---

## License

MIT
