# Interface Runtime Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Verify the proposed Node, npm and Astro runtime in disposable hosted execution, retain an integrity-bearing candidate dependency lock and license declarations, and compare two clean offline synthetic builds.

**Architecture:** A trusted hosted download stage verifies fixed primary archives and an immutable Linux/amd64 container image. One restricted networked container resolves and provisions registry dependencies with lifecycle scripts disabled; a second restricted container denies network and performs two clean installs and builds from that frozen lock and warmed cache. This is preparation compatibility, with no actual reader input or feature RED/GREEN claim.

**Tech Stack:** Node24.21.0, npm11.19.0, Astro7.3.8; official Node bookworm-slim image; npm integrity cache; Ubuntu24.04 Docker; Python stdlib data inspection; hosted Ruby Psych YAML parsing.

**Spec:** [Interface plan Task2](2026-10-08-architecture-interface.md) and the [separate first public-projection unit](2026-10-08-artifact-index-contract.md). The interface capture contract is read for exclusion boundaries; no private capture, policy, real public projection or graph is consumed here.

## Global Constraints

- Scoped amendment main307/6070555716 extends interface pickup6070431864. This author owns only .github/workflows/interface-runtime-preparation.yml, interface/package.json and this plan. Other source/tests, the actual lockfile path, existing workflows, formal producers, private capture and peer site/browser scopes remain outside this amendment.
- Standing owner authorization supplies the operational permission. Root publishes the frozen source after nonauthor source review and conducts actual hosted execution; this author executes no project Python, Node, npm, Docker container, package installation or build on the laptop.
- Scientific effect is NONE; scientific status authority is exact false. Packet custody, mathematical alignment, scientific acceptance and reader feature completion are not established by these smoke checks.
- The checked source file mounted into either container is only interface/package.json, read-only. The immutable generated control and fixture mounts are also read-only; no capture, evidence adapter original_packet, Artifact Index, graph, teaching source or existing site file is mounted.
- This additive workflow uses pull_request path filters and workflow_dispatch, read-only contents permission and no credentials or deployment action. It does not alter the existing required verify/formal/architecture DAG.
- Preserve failure and setup diagnostics, actual status files, original runtime/package bytes and strict cleanup outcome. A missing receipt, failed preparation, collection refusal, skipped stage or cleanup failure is not compatibility success.

## Primary pins and observed metadata

Primary data was read on8October2026; no candidate was installed. Fixed package metadata and archive integrity strings are:

| Input | Identity | Primary pin / declared license |
|---|---|---|
| Node Linux x64 tar.gz |24.21.0 | SHA2566e1db87ef58b8819e5d5402eff1536491b18edd8eb7bee5ef7897876e88dc5ff |
| npm archive |11.19.0 | sha512-SDd/hHg3KqHE5Ht2NHWxNYNtqCQ2pXAPLl6OtQhPyED5PHsRfrOtO199MZTIG2cQoQ1ZRI9t28shrD+2cr3AAw==; Artistic-2.0 |
| Astro archive |7.3.8 | sha512-LHzDHSaV+YzZiR3EFndDYTLLeNeuMtP1qX1FLcCNUUHOCqTTIWPczy2UV3zpA5VqvJe5cWZolqM9Z8JFqR2i9w==; MIT |
| Node OCI manifest | Linux/amd64 | sha256:51b1100cc2a83d370c6a60952e3f2989c8a43159d0e38586e090f3b3326efefd |
| Node OCI config | Linux/amd64, NODE_VERSION24.21.0 | sha256:470015c1ab1577c19efce2e80cdad4afd9c682cf8f4dcf3dccae5abfd9113397 |

The OCI manifest raw bytes independently hash to the manifest digest above. Its official source annotation points to nodejs/docker-node revision93a7bafc324a85ac1ee461604cff87cffacb6d7a,24/bookworm-slim. The corresponding Dockerfile installs Node24.21.0 and retains its dynamically required libraries. This avoids assuming that a Python slim image has compatible libstdc++/glibc. The hosted run still must establish that the separately SHA-pinned Node archive binary actually runs against this image and Astro's native dependencies; no successful runtime compatibility is inferred from metadata. Independently streamed original npm and Astro tarball bytes match the fixed SHA512 strings above; reading/hashing those archives did not install or execute them. The npm archive contains only regular members, with the actual LICENSE and bin/npm-cli.js present.

Astro metadata declares Node>=22.12.0, npm>=9.6.5 and MIT. npm metadata declares Node^20.17.0 or >=22.9.0 and Artistic-2.0. All three are exact candidates in package.json. The workflow verifies original downloaded Node SHA256 and package SHA512/SHA1 against these independent fixed pins before extracting runtime bytes. Mutable metadata can cause a refusal, not select another version or archive.

Primary references: [Node checksum file](https://nodejs.org/dist/v24.21.0/SHASUMS256.txt), [Astro exact metadata](https://registry.npmjs.org/astro/7.3.8), [npm exact metadata](https://registry.npmjs.org/npm/11.19.0), [official image manifest](https://registry-1.docker.io/v2/library/node/manifests/sha256:51b1100cc2a83d370c6a60952e3f2989c8a43159d0e38586e090f3b3326efefd), [official Dockerfile](https://github.com/nodejs/docker-node/blob/93a7bafc324a85ac1ee461604cff87cffacb6d7a/24/bookworm-slim/Dockerfile), [Astro configuration](https://docs.astro.build/en/reference/configuration-reference/), [npm ci](https://docs.npmjs.com/cli/v11/commands/npm-ci/).

## Review Focus

- Exact direct versions do not freeze transitive dependencies. The first network resolution produces a provisional lock; validate every registry URL and integrity, use that one lock in both offline installs, retain its exact bytes and review before a later owner claims and commits interface/package-lock.json.
- Metadata compatibility does not prove dynamic-library or native-binding compatibility. Execute the original pinned Node binary and pinned npm CLI in the immutable image, check actual versions, and require both Astro compilations.
- Registry dependency scripts or a build running while networking remains available would cross the preparation boundary. Disable installation lifecycle scripts in every stage; run builds only in the distinct network-none container.
- A directory diff alone can miss a wrong publication base or inline stylesheet. Require emitted base/canonical links, an actual external CSS member containing the fixture marker, no inline style/script, and identical complete file/byte/hash inventories.
- A timed-out container or symlinked output must not remain a writer during collection. Dispose both named containers, recheck absence, and open bounded retained files through anchored no-follow descriptors before upload.

---

## Task1: Hosted provisioning and synthetic offline compatibility

**Files:** Create .github/workflows/interface-runtime-preparation.yml, interface/package.json and this plan only.

**Interfaces:** The workflow consumes the exact checked package source plus the fixed public runtime/package downloads. It produces one native artifact named interface-runtime-preparation-RUN_ID-RUN_ATTEMPT containing the provisional package-lock.json, declared license inventory, original Node/npm/Astro archives and metadata, runtime/license identities, immutable controls, two output trees and manifests, logs/status files, collection report, checksums and a preparation-only receipt when every check succeeds.

- [ ] Freeze these three source files and obtain nonauthor source review, including real YAML parsing in the hosted Ubuntu24.04 Ruby3.2/Psych runtime. Root publishes the exact reviewed source via its available native Git capability.
- [ ] Run the preparation workflow on that exact candidate. Root records actual checked commit, repository, run/attempt, job result and native artifact ID/name/digest. No absent reader CLI or smoke setup failure is called feature RED.
- [ ] Verify fixed downloads, their recorded engines/licenses and actual image inspect identity. Runtime extraction admits only the fixed Node binary/LICENSE and bounded regular npm members under its package prefix. Retain original archives even when a later compatibility step fails.
- [ ] Resolve the one exact Astro dependency with the verified npm11.19.0 CLI, using package-lock-only and lifecycle scripts disabled. Admit at most1024 lock package entries; every nonroot entry must have an exact version, SHA512 integrity and an HTTPS registry.npmjs.org tgz URL without credentials/query/fragment. Refuse Git, file, custom-registry and unbound entries before downloading the dependency closure.
- [ ] Retain the original checked package, exact validated generated lock, source hash and declared license inventory before networked npm ci can fail or time out; then provision with lifecycle scripts disabled and strict engine enforcement. Inventory entries describe the complete declared lock, including optional foreign-platform packages; they do not claim every optional package installed. Missing license declarations remain UNKNOWN; legal approval remains NOT_ASSESSED.
- [ ] Dispose the networked stage automatically or explicitly on timeout, then run the distinct network-none container. For each clean work directory, copy only the immutable synthetic Astro fixture, checked package and one frozen lock; run npm ci --offline --ignore-scripts and npm run build with pinned runtime commands.
- [ ] Require actual Node24.21.0/Linux/x64, npm11.19.0 and installed Astro7.3.8. The smoke configuration is site https://d6g8k5htny-coder.github.io, base /main/reader, static output and inlineStylesheets never. Test actual emitted base/canonical links, external compiled stylesheet/marker and absence of inline style/script. Compare both complete output manifests including membership, size and SHA256.
- [ ] Dispose both containers before trusted collection. Every Docker list query uses a standalone checked assignment: a daemon error cannot be interpreted as an empty list. An isolated privileged stdlib collector reads UID65532-private directories through anchored no-follow descriptors only after checked absence of both containers. It creates retained files through anchored exclusive no-follow descriptors and assigns those created files back to the host UID/GID; it never recursively changes ownership or follows product links. Independently rehash every admitted output file, require equal retained manifests and check original host status/context/image/package identities. Produce the preparation-only receipt without claiming actual reader feature or source custody.
- [ ] Upload bounded evidence on success or failure, then strictly unmount the output filesystem. Native job success includes cleanup. A receipt alone cannot override failed/skipped cleanup or archival stages.
- [ ] Download the original native artifact, verify its original ZIP digest and all retained member hashes, and report exact compatibility outcome and residual unknowns through root's existing evidence surfaces. Only then claim the candidates were installed/tested in this hosted preparation.

## Resource and evidence bounds

The shared scratch filesystem is2GiB ext4 with nodev/nosuid; executable Node/npm/native dependencies require exec. Only the work subtree is writable by UID/GID65532. Runtime/control/source mounts are read-only, container roots are read-only, all capabilities are dropped and no-new-privileges is set. Both containers have2 CPUs,2GiB RAM,256 PIDs,128MiB per-file hard limit and64MiB noexec/nosuid/nodev tmpfs. The provisioning container permits network solely for npm resolution/download; validated lock URLs restrict its selected packages to the primary registry, but this is not a network egress firewall. The separate compile container uses Docker network none.

The job timeout is35minutes, leaving collection/upload/cleanup time beyond the sequential download and container caps. Image pull180seconds, provisioning420seconds and offline execution360seconds each have a TERM timeout with a10second KILL fallback. npm operations have additional subprocess timeouts. Host console files use an8MiB file limit and package/runtime downloads have explicit connection/time/byte limits. Lockfiles are bounded at16MiB, individual retained logs/manifests at4MiB, each build at128 files/16MiB total and bounded collection at192MiB. Unknown or oversized outputs refuse collection rather than being followed or silently treated as success. Original failure consoles remain in the trusted publication directory. No raw container logs are printed as GitHub workflow commands. Cancellation or whole-job timeout can prevent final retention/cleanup steps; such a run remains failed or unresolved, never compatibility acceptance.

The artifact receipt binds actual native checked commit, repository, run and attempt plus a complete member byte/SHA inventory. It explicitly requires native job success and original archive readback. The uploader's native archive identity remains an external root readback responsibility; no inner receipt or copied digest is authenticated custody.

## Current disposition and later work

At author freeze: SOURCE_PREPARED; installed NOT_RUN; dependency resolution NOT_RUN; two offline compilations NOT_RUN; actual reader feature RED/GREEN NOT_RUN; original native archive readback NOT_RUN. Candidate metadata/pins have been read, not installed or tested. Root owns source review, publishing and execution.

After a successful observed preparation, root must separately claim/review the generated lockfile for committed reproducibility and dependency/license disposition. The reader source/public input loader, graph/archive custody, mathematical teaching models, browser verification, preservation of existing namespaces, Pages publication, true500-token retrieval and remaining architecture goals remain unfinished. This workflow is additive preparation and does not replace any existing required check or peer review.

## Rejected source checkpoint and correction

The initial runtime workflow SHA256e02e34a1374bebe5735efab8179407702c74c0bec91cc7043b3316229754a5c7, with plan SHA2562764d2ad6e43935dd36e7f208de9b0f991302edbbe17ce437b774878108f3dab and unchanged package SHA2565edbc4a2a3e75fd9b9c0de99c52390e38316c2bb8885e6568c70a0829d1ee5d4, was rejected in distinct nonauthor source review before publication/execution. Its four blockers were:

- Workflow37–38 transferred work-directory ownership to UID65532 before the unprivileged host attempted chmod0700; that host chmod would fail.
- Workflow355–359 later attempted unprivileged collection through that mode0700 UID65532 directory; the host could not traverse it.
- Workflow209–219 wrote the validated lock and license/source evidence only after npm ci succeeded; a failed installation would lose the already resolved candidate from the retained artifact.
- Workflow337–344 nested docker ps inside emptiness tests; a daemon error returning empty stdout could masquerade as confirmed absence and admit collection.

The amended source orders chmod before chown, uses an isolated privileged anchored data collector with host-owned exclusive output files, retains the validated candidate before installation, and propagates each Docker query failure before any absence test. These are source corrections, not observed hosted RED/GREEN or cleanup success. The rejected bytes remain historical evidence; amended-source acceptance, hosted YAML/runtime execution and original native archive readback remain pending. No laptop project execution, installation, publication or other scoped source edit occurred.
