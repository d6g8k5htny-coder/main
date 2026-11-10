# Initial predecessor-read orchestration failure

Read-only run `42758305250`, job `129637755016`, failed before any public readback: isolated Python could not import `current_site_data` because the temporary workflow added the repository root rather than its `tools` directory to `sys.path`. The original workflow and complete failure log are retained. This is an orchestration failure, not evidence about public artifact bytes, the receipt, deployment cancellation or recovery. The follow-up observer fixes only that import path and uses the same pinned Main source and receipt guards.
