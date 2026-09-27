# Changelog

Notable changes to PETScAgent-Bench are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Framework-neutral Purple Agent efficiency reporting at the A2A boundary.
- Optional `petscagent.telemetry.v1` data for model calls, tool calls, token
  usage, peak context size, and cost.
- A separate budget-based efficiency score that does not affect the existing
  composite quality score or tier.
- Time-to-first-response and response-event measurements using A2A client
  interceptors.
- Protobuf response caching keyed by Purple model and problem.

### Changed

- Migrated from A2A SDK 0.3 to 1.x.
- Migrated from FastMCP 2.x to 3.x.
- Replaced the `dotenv` shim with `python-dotenv` and removed the obsolete
  `pathlib` backport.
- Updated compatible direct and transitive dependencies.

### Compatibility

- Legacy A2A 0.3 `.pkl` response caches are not reused; new caches use the
  A2A 1.x protobuf `.pb` format.
- FastMCP 4 is deferred because the upstream PETSc MCP server does not yet
  support its reorganized tool API.
- OpenAI 3 is deferred because the current LiteLLM release requires OpenAI 2.

## 1.0.0 - 2026-09-26

- Established the published-paper implementation as the first stable release.
