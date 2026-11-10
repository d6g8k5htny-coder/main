"""One-shot same-repository rerun request for a single observed failed run."""
import json
import os
import urllib.request

REPOSITORY = 'd6g8k5htny-coder/main'
RUN_ID = 42778804824
HEAD = 'b285dd7a31ad7b297abc9f2aa93267300111a4e9'
REF = 'refs/heads/codex/cbz-architecture-observation-20261110'
API = 'https://api.github.com'


def guard_environment(env):
    expected = {'GITHUB_REPOSITORY': REPOSITORY, 'GITHUB_REF': REF,
                'GITHUB_EVENT_NAME': 'push', 'GITHUB_API_URL': API}
    if any(env.get(k) != value for k, value in expected.items()):
        raise ValueError('unexpected runner context')
    token = env.get('GITHUB_TOKEN')
    if not isinstance(token, str) or not token:
        raise ValueError('missing workflow token')
    return token


def guarded_rerun(run, send):
    expected = {'id': RUN_ID, 'head_sha': HEAD, 'run_attempt': 2,
                'status': 'completed', 'conclusion': 'failure', 'event': 'push',
                'path': '.github/workflows/workspace-landing.yml'}
    if not isinstance(run, dict) or any(type(run.get(k)) is not type(value) or run.get(k) != value
                                        for k, value in expected.items()):
        raise ValueError('observed run differs from authorized target')
    repo = run.get('repository')
    if not isinstance(repo, dict) or repo.get('full_name') != REPOSITORY:
        raise ValueError('observed repository differs from authorized target')
    return send()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirect refused')


def main():
    token = guard_environment(os.environ)
    opener = urllib.request.build_opener(NoRedirect())
    url = API + '/repos/' + REPOSITORY + '/actions/runs/' + str(RUN_ID)
    headers = {'Accept': 'application/vnd.github+json', 'Authorization': 'Bearer ' + token,
               'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'cbz-one-shot'}
    with opener.open(urllib.request.Request(url, headers=headers), timeout=30) as response:
        if response.status != 200:
            raise ValueError('run lookup did not succeed')
        raw = response.read(1048577)
        if len(raw) > 1048576:
            raise ValueError('run response exceeded limit')
    run = json.loads(raw)

    def send():
        request = urllib.request.Request(url + '/rerun', data=b'{"enable_debug_logging":false}',
                                         headers={**headers, 'Content-Type': 'application/json'}, method='POST')
        with opener.open(request, timeout=30) as response:
            if response.status != 201:
                raise ValueError('rerun request not accepted')
            response.read(4096)
        print('Full rerun accepted for run 42778804824 after exact attempt-2 guard')

    guarded_rerun(run, send)


if __name__ == '__main__':
    main()
