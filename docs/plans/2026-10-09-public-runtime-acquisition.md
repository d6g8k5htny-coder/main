# Public runtime acquisition preparation contract

Status: SOURCE_PREPARED; hosted acquisition, probes, recovery and archive custody
NOT_RUN at this freeze. Scope is native issue 307 comment 6089311554: this new
contract and `.github/workflows/architecture-runtime-acquisition.yml` only.
Neither an author inspection nor successful preparation changes an existing
producer, receipt, product, test inventory or scientific status.

## Purpose and evidence boundary

Two original hosted attempts stopped at the Docker Hub unauthenticated pull
rate limit: routing 37990894091/job 114024452509 and adapter 37990676085/job
114023732707. No architecture test ran. These are acquisition failures, supply
no behavioral RED or GREEN, and do not authorize the pending package fix.
The failed originals and missing observations remain historical evidence.

The exact canonical reference is
`python:3.11.14-slim-bookworm@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48`.
The required unmodified native `RepoDigests` value is the singleton
`["python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"]`.
The configuration/ImageID is
`sha256:44d8f4434bcb025b8cb5c3151be689a906223e47cc3146971feccda936d2e82b`.
The preparation compares those values with independently fetched original
manifest/configuration bytes and actual Docker/runtime observations.

An unauthenticated public metadata GET returned HTTP 200 and a 1,752-byte
single OCI image manifest whose raw SHA-256 is the exact 83f339 digest. Its
configuration descriptor identifies 5,499 bytes with the exact 44d8 digest;
independent inert configuration read/hash confirms linux/amd64,
`PYTHON_VERSION=3.11.14` and four ordered `rootfs.diff_ids`. Its four compressed
layer descriptors total 47,685,311 bytes. These observations establish metadata
byte compatibility only. No local image layer was downloaded or executed; a
hosted Docker acquisition has not been established by these HTTP observations.

The [OCI manifest specification](https://github.com/opencontainers/image-spec/blob/main/manifest.md)
distinguishes a single image manifest from a platform index and links a
configuration and ordered layers by digest. The [OCI configuration specification](https://github.com/opencontainers/image-spec/blob/main/config.md#imageid)
defines ImageID as SHA-256 of configuration JSON. This supports comparing 44d8
with the native Docker Id; actual Docker observations still have to agree.

## Mechanism and current consumers

Use the official Docker daemon registry-mirror mechanism on a fresh disposable
GitHub-hosted ubuntu-24.04 runner, preserving the canonical image reference.
Google documents `registry-mirrors: ["https://mirror.gcr.io"]`, a daemon restart,
and checking actual Docker info. Its cache can evict images; a cache miss falls
back to Docker Hub. Cache availability and a successful hosted pull are not
guaranteed. Acquisition origin and fallback remain `USED_UNKNOWN`, including
after successful preparation. See [Google's public-cache instructions](https://docs.cloud.google.com/artifact-registry/docs/pull-cached-dockerhub-images)
and [Docker's mirror instructions](https://docs.docker.com/docker-hub/image-library/mirror/).
[Docker's digest-pull documentation](https://docs.docker.com/reference/cli/docker/image/pull/)
supports the immutable digest reference and explicit linux/amd64 selection.

The frozen PR #325 generic-workflow candidate at `6d024e8b` is 183 lines, SHA-256
`5ce726b9de40e9decc230c5429d224a5ca33dc64af458cea495334954b457652`.
Separately, current native main at `ecedd604` has a 183-line, 10,730-byte generic
workflow, SHA-256
`75867d388dede2a9840c1efc455449bc5531a14a023b4edb2d07aab3541e2458`,
with native `actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9`.
Any future generic-workflow migration preserves that current main upload pin.
Its host receipt is `tools/architecture_run_receipt.py`, 354 lines, SHA-256
`a4bb66325f64422398ff726bc1c448afb284498275e5894a7d1af6b25fba9890`;
receipt controls are 695 lines, SHA-256
`56b01ae5ae5faf37675ee84771d81cb81a13c2224658067d37220724013fc84a`.
Receipt line 258 requires the exact singleton canonical RepoDigests; it accepts
and preserves any correctly formatted native ImageID, rather than pinning 44d8.
Its existing positive control explicitly permits a different valid ImageID.
The current routing workflow is 1,343 lines, SHA-256
`ec55402f53bbde7681e9f581356b558e61502b5c6c3f4ebbd1c23174b1334971`;
lines 939–944 require the canonical RepoDigest and exact configured reference.
The receipt and receipt controls are unchanged between that candidate and
current main. This preparation edits none of those sources. Future generic,
routing, semantic or lease
migration needs its own exact-path claim and complete source/control review.

A direct `mirror.gcr.io/library/python@sha256:83f339…` pull is WATCH: a native
mirror-prefixed RepoDigest would fail the current canonical receipt predicate.
[Docker tagging](https://docs.docker.com/reference/cli/docker/image/tag/)
creates a tag alias; it is not evidence of a canonical repository digest.
[Docker load](https://docs.docker.com/reference/cli/docker/image/load/)
documents restored images and tags, without promising the required RepoDigests.
Neither rewriting inspect JSON nor manufacturing that field is permitted.

## Frozen external preparation controls

These controls define observable behavior before the workflow is written. All
are selected together. They are public, source/test exposed, and are not an
organizationally independent or authenticated observation channel.

| ID | Required observation and refusal predicate |
| --- | --- |
| RP01 | Actual repository/run/attempt/checked commit, full Git tree/parents, original two source files and trusted controller/probe bytes. The checked worktree equals the native commit; both source byte/hash records remain equal before and after. |
| RP02 | Original daemon-file presence, regular-file identity/mode/owner/bytes, full admitted daemon JSON, original Docker info/mirrors/service state, empty **all-container** inventory and native **all-image**, including intermediate-image, inventory. Any existing container or matching 44d8 image content or 83f manifest prevents fresh-acquisition acceptance. This is not an absence claim for every cached layer. |
| RP03 | A fresh empty 0700 DOCKER_CONFIG is the sole client configuration. Closed typed public daemon admission precedes every daemon original/raw/request capture. Temporary daemon JSON preserves every admitted original setting except the mirror entry. One actual restart succeeds; actual Docker info reports only the public mirror. Unsupported keys, nested options, credentials and proxies yield metadata-only refusal before mutation; their values and raw bytes are not published. |
| RP04 | One bounded unauthenticated HTTPS GET per exact manifest/configuration URL; original successful response headers and bytes, exact media types, byte counts, SHA-256, configuration pointer, four ordered layer descriptors and configuration diff IDs. Metadata mismatch or incomplete capture refuses acceptance. |
| RP05 | Exactly one bounded canonical linux/amd64 Docker pull; full admitted stdout/stderr, exit status, timeout/completeness and actual Docker client/server versions. Nonzero, timed-out, truncated or missing acquisition refuses acceptance. No pull loop, login or manual token acquisition. |
| RP06 | One unmodified native image inspect: exact singleton canonical RepoDigests, Id44d8, linux/amd64, Python environment and ordered native RootFS layers equal the independently captured configuration. Extra aliases or forged normalization refuse acceptance. |
| RP07 | Three newly created distinct owned containers, one each for 0/O/OO. Actual before/after container identity, image/reference, entrypoint/argv/environment/mounts and kernel restrictions agree with the fixed trusted probe. No project source or fixture mount and no implicit pull: creation uses `--pull=never`. |
| RP08 | All three complete probes report Python 3.11.14, SQLite 3.40.1, exact interpreter flags, executable path, actual SQL settings/STRICT table/rollback and complete compile-option inventory. Hash/size/path observations of Python executable, sqlite module, sqlite extension, loaded sqlite library and loaded Python library agree across modes. |
| RP09 | Actual UID/GID65532, CapEff0, NoNewPrivs1, root/control read-only, only loopback/no external routes, actual 2 CPU/1 GiB/128 PID limits, 256 descriptors and 64 MiB FSIZE. Full original mountinfo, every parsed mount and each mountpoint's identity/type/permissions/capacity are retained. The writable application work area is /tmp, a 64 MiB noexec/nodev/nosuid tmpfs. Explicit Docker exceptions are bounded 1 MiB /dev/shm, device tmpfs /dev (at most 64 MiB, root-owned directory unwritable by UID65532), devpts, mqueue with actual 8192-byte RLIMIT_MSGQUEUE, proc kernel interfaces, closed Docker-masked proc null-device bindings, and Docker hosts/hostname/resolv files (root-owned, at most 64 KiB each, unwritable by UID65532). Their native types/options/permissions/capacities are checked; sys/cgroup and every other mount are read-only. Device, proc and IPC interfaces are declared kernel exceptions, not additional application data directories. No hidden writable data mount is admitted. |
| RP10 | All three owned containers reach a complete zero-exit non-OOM terminal state and are confirmed absent. Only actually created container IDs may be disposed. Existing unrelated containers are never stopped/removed. Disposal precedes restoration and receipt acceptance. |
| RP11 | Original daemon-file bytes, presence, UID/GID/mode or original absence are restored; one restoration restart and actual Docker mirror state agree with the original baseline. Changed ownership/configuration, unexpected active containers or any incomplete recovery prevents acceptance. |
| RP12 | Closed regular-file originals, bounded unchanged reads, source before/after joins, all command/probe records, recovery, receipt and original byte hashes retained before upload. Missing, refused, incomplete or oversized expected originals prevent acceptance. Artifact/native raw-byte readback is required after the job, outside the workflow's acceptance claim. |

The hosted semantic refusal controls operate on copied actual observations,
leaving originals unchanged. They call the same acceptance predicate used by
the receipt and preserve all unrelated observations. They are labeled TEST,
not runtime acquisition or independent scientific evidence. Required cases:

1. RP01 stale run identity; RP01 changed after-source hash.
2. RP02 preexisting container; RP02 preexisting exact image.
3. RP03 actual mirrors empty; RP03 changed preserved daemon setting;
   RP03 nested `log-opts/splunk-token` synthetic credential in copied actual data.
4. RP04 changed manifest digest; RP04 changed configuration digest.
5. RP05 nonzero pull; RP05 incomplete pull capture.
6. RP06 mirror-prefixed digest; RP06 extra digest; RP06 wrong ImageID/platform.
7. RP07 changed native UID; RP07 missing OO container/probe.
8. RP08 wrong SQLite version in all modes; RP08 wrong OO optimization;
   RP08 empty binary hash in all modes.
9. RP09 wrong CPU quota in all modes.
10. RP10 unconfirmed disposal; RP11 wrong restored mirror state;
    RP11 different restored daemon bytes; RP12 missing original.

The unmodified copied actual observation is a required positive control. Each
listed negative must be refused; any unexpected acceptance fails preparation.
All original 23 negative IDs remain selected, with the nested-credential case
as the 24th negative. A separate early synthetic daemon-admission control also
requires a typed public positive and nested `splunk-token` refusal before any
daemon raw capture. It does not substitute for the copied-actual positive.
Absent actual observations leave these controls NOT_RUN rather than inventing
a RED. A first genuine successful preparation is allowed without a fake RED.

## Bounds, preparation and recovery plan

Only this contract and the new standalone workflow are repository edits. Freeze
this contract first; embed its byte/hash pin in the workflow. Write the fixed
probe and trusted host controller as literal workflow sources, then have a
distinct nonauthor read both complete frozen sources before root publication.
No local Python, Docker, SQL, YAML parser, project or model execution is allowed
for this source-only authoring task. The hosted parser/probes remain NOT_RUN.

The hosted job has a 20-minute outer limit. Each ordinary native observation is
bounded to 30 seconds, service restart to 60 seconds, the single pull to 180
seconds and each probe attachment to 60 seconds. One termination grace is at
most 5 seconds. Each original stream/file is at most 1 MiB except pull streams,
which are at most 4 MiB each; original daemon bytes are at most 64 KiB and each
public JSON response at most 64 KiB. The retained inventory has at most 256
members and at most 32 MiB of admitted original bytes; the final manifest and
receipt have separate 1 MiB caps and the whole upload is capped at 36 MiB.
Those are defensive ceilings, not predicted timings or physical image sizes.
Image layers are acquired by hosted Docker only and are not archived locally
or included in the retained-output subtotal.

Capture native context and source originals before preparation. Refuse a
nonempty container inventory, preexisting exact image or unsafe/private daemon
configuration before configuration changes. Retain the original file or its
observed absence. Apply the one temporary mirror configuration, restart and
observe actual state; capture independent public metadata, pull once, inspect
and run the three fixed probes. Dispose each owned container promptly and again
check the complete native container inventory before recovery. Restore only
the owned temporary daemon file while its bytes/identity still match. Restore
the original file identity policy or absence, restart and observe the original
mirror state. Recovery runs after failure too; acceptance waits for recovery.

On failure, retain every admitted available original and exact missing/refused
inventory. Unexpected credential/private configuration is not an admissible
public original: retain only its presence/hash/refusal metadata, mark setup
blocked, and do not publish its contents or change the daemon. No partial
capture, recovery failure or absent controls can emit ACCEPTED_PREPARATION.
Upload admitted originals on every conclusion, then remove only this job's
owned temporary client/control files. Native archive hash/member-copy readback
and complete distinct review still precede any producer migration.

Disposition: ADAPT for this controlled hosted daemon-mirror experiment; actual
acquisition/runtime remain UNPREPARED until measured. Success is preparation
evidence only. Acquisition origin/fallback USED_UNKNOWN; source/test exposure
true; organizational independence 0; scientific effect NONE;
scientific-status authority false; custody authentication UNKNOWN.

Source-only appendix before final workflow pin freeze: lease preparation
37991189760/job 114025477625 also stopped at the same unauthenticated pull-rate
limit. It supplies no runtime or behavioral RED/GREEN. The independent local
configuration body/hash observation did not retain its transport headers:
local configuration HTTP Content-Type remains UNKNOWN. The hosted single GET
retains the actual transport headers; only the manifest HTTP and OCI descriptor
media types are pinned. Blob transport MIME is observed without assuming it
equals the OCI configuration descriptor's MIME.

The controller reserves a 720-second preparation phase, a 120-second owned
disposal allowance when leaving a probe, and a 240-second recovery phase within
the outer job ceiling. Per-command limits can shorten at those deadlines.
Controlled command termination uses at most four seconds before SIGKILL and
one second to reap. Controller admission is at most 30 MiB, reserving space for
the closed seal/source fallbacks and its at-most-1-MiB collection metadata;
their combined admitted originals remain at most 32 MiB. A failed capture can
never earn acceptance, but does not prevent an otherwise observed owned
physical disposal/restoration from continuing. Platform termination can leave
recovery unobserved; that remains SETUP_BLOCKED, never accepted preparation.

Review correction under renewed scope 307/6089713967: the public daemon schema
admits only boolean debug/live-restore/network switches; typed mtu/log-level;
the runc default runtime; fixed storage-driver enum; absolute data-root;
public HTTPS registry-mirrors; native cgroup-driver exec options; two boolean
feature flags; and local/json-file/none log drivers with closed max-size,
max-file, compress, mode and max-buffer-size options. Every other key or nested
option is unsupported and returns only presence/identity/byte-hash/refusal
metadata. This includes `log-opts/splunk-token`, regardless of its value.
Termination and cleanup exceptions are recorded without escaping partial
stream capture. Unreaped process uncertainty refuses acceptance, has no retry,
and remains visible in the command status. Full mount observations replace the
earlier incorrect sole-writable-filesystem claim. All four review findings and
the 186-line/1094-line pre-review freezes remain retained historical records.

Continuation scope 307/6110587035 on 11 October preserves the interrupted
205-line contract and 1196-line workflow as historical, unreviewed partial
corrections. Complete mount validation explicitly includes proc and only
Docker's closed masked proc null devices, whose native device identity is
major/minor 1:3 and whose paths occur in actual HostConfig.MaskedPaths. Full
mountpoint facts are observations, not claims that virtual kernel filesystems
have ordinary disk quotas. devpts/proc remain subject to the observed PID,
descriptor and cgroup limits; mqueue also has the measured per-UID byte limit.
The [Moby mount implementation](https://github.com/moby/moby/blob/master/daemon/oci_linux.go)
identifies the proc/dev/IPC exceptions to root read-only handling. The
[kernel mountinfo documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html#proc-pid-mountinfo-information-about-mounts)
distinguishes per-mount options from superblock options; admission uses the
former, retaining both. Parent IDs outside the namespace are not contradictions.
An unreaped native command may still be mutating the daemon or creating a
container. Further container/daemon recovery operations are refused in that uncertain
state, retaining SETUP_BLOCKED and unobserved recovery rather than racing it.
