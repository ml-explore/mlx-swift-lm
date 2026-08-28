"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { STATE_LABELS, RUNNING, onStart } = require("./ci-state.js");

function step(name, conclusion) {
  return { name, conclusion };
}

function job(name, conclusion, steps = []) {
  return { name, conclusion, steps };
}

test("STATE_LABELS names the seven labels this script manages", () => {
  assert.deepEqual(STATE_LABELS, [
    "needs-scan",
    "needs-ci",
    "needs-lint",
    "needs-changes",
    "needs-review",
    "approved",
    "ready-to-merge",
  ]);
});

test("needs-rebase is not managed here", () => {
  assert.equal(STATE_LABELS.includes("needs-rebase"), false);
});

test("RUNNING is ci-running", () => {
  assert.equal(RUNNING, "ci-running");
});

test("onStart adds ci-running", () => {
  assert.deepEqual(onStart([]), { add: ["ci-running"], remove: [] });
});

test("onStart removes the labels that describe the previous run", () => {
  const result = onStart(["needs-ci", "cat-model"]);
  assert.deepEqual(result.add, ["ci-running"]);
  assert.deepEqual(result.remove, ["needs-ci"]);
});

test("onStart keeps approved, ready-to-merge and needs-scan", () => {
  const result = onStart(["approved", "ready-to-merge", "needs-scan", "needs-review"]);
  assert.deepEqual(result.remove, ["needs-review"]);
});

test("onStart never removes a label outside its list", () => {
  const result = onStart(["unread", "cat-server", "changes requested"]);
  assert.deepEqual(result.remove, []);
});

test("onStart does not re-add ci-running when it is already held", () => {
  assert.deepEqual(onStart(["ci-running", "cat-model"]), { add: [], remove: [] });
});

const { CONTRACT, CONTRACT_PREFIX, contractOf } = require("./ci-state.js");

test("CONTRACT is a whole number above zero", () => {
  assert.equal(Number.isInteger(CONTRACT), true);
  assert.equal(CONTRACT > 0, true);
});

test("CONTRACT_PREFIX names the step prefix", () => {
  assert.equal(CONTRACT_PREFIX, "Contract: ");
});

test("contractOf reads the number from a step that passed", () => {
  const jobs = [job("ci_state_script", "success", [step("Contract: 4", "success")])];
  assert.equal(contractOf(jobs), 4);
});

test("contractOf finds the marker in any job", () => {
  const jobs = [
    job("lint", "success", [step("Run style checks", "success")]),
    job("ci_state_script", "success", [step("Contract: 7", "success")]),
  ];
  assert.equal(contractOf(jobs), 7);
});

test("contractOf returns null when no step names a number", () => {
  const jobs = [job("lint", "success", [step("Run style checks", "success")]), job("x", "success")];
  assert.equal(contractOf(jobs), null);
});

test("contractOf returns null for text that is not digits", () => {
  assert.equal(contractOf([job("ci_state_script", "success", [step("Contract: two", "success")])]), null);
  assert.equal(contractOf([job("ci_state_script", "success", [step("Contract: ", "success")])]), null);
});

const { onComplete } = require("./ci-state.js");

const GREEN = [
  job("lint", "success", [step("Run style checks", "success")]),
  job("mac_build_and_test", "success", [step("Build (Xcode, macOS)", "success")]),
];

test("everything green with no approval gives needs-review", () => {
  const result = onComplete({ conclusion: "success", jobs: GREEN, labels: ["ci-running"] });
  assert.deepEqual(result.add, ["needs-review"]);
  assert.deepEqual(result.remove, ["ci-running"]);
});

test("green on an approved pull request gives ready-to-merge", () => {
  const result = onComplete({
    conclusion: "success",
    jobs: GREEN,
    labels: ["ci-running", "approved", "cat-model"],
  });
  assert.deepEqual(result.add, ["ready-to-merge"]);
  assert.deepEqual(result.remove, ["ci-running", "approved"]);
});

test("green on a pull request already ready-to-merge writes nothing new", () => {
  const result = onComplete({
    conclusion: "success",
    jobs: GREEN,
    labels: ["ci-running", "ready-to-merge"],
  });
  assert.deepEqual(result.add, []);
  assert.deepEqual(result.remove, ["ci-running"]);
});

test("a skipped job is not a failure", () => {
  const jobs = [...GREEN, job("integration_build_xcode27", "skipped")];
  const result = onComplete({ conclusion: "success", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-review"]);
});

test("no failed job and a conclusion other than success asks for another run", () => {
  const result = onComplete({ conclusion: "cancelled", jobs: GREEN, labels: ["ci-running"] });
  assert.deepEqual(result.add, ["needs-ci"]);
});

test("a failed job with no failed step asks for another run", () => {
  const jobs = [job("mac_build_and_test", "failure", [step("Build (Xcode, macOS)", "success")])];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-ci"]);
});

test("an infrastructure step failing alone asks for another run", () => {
  const jobs = [
    job("lint", "success"),
    job("mac_build_and_test", "failure", [step("Infra: verify MetalToolchain installed", "failure")]),
  ];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-ci"]);
});

test("an unprefixed infrastructure step also asks for another run", () => {
  const jobs = [
    job("lint", "success"),
    job("mac_build_and_test", "failure", [step("Verify MetalToolchain installed", "failure")]),
  ];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-ci"]);
});

test("an unprefixed infrastructure step alongside a real failure blames the author", () => {
  const jobs = [
    job("mac_build_and_test", "failure", [
      step("Install MetalToolchain", "failure"),
      step("Build (Xcode, macOS)", "failure"),
    ]),
  ];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-changes"]);
});

test("an infrastructure step alongside a real failure blames the author", () => {
  const jobs = [
    job("mac_build_and_test", "failure", [
      step("Infra: verify MetalToolchain installed", "failure"),
      step("Build (Xcode, macOS)", "failure"),
    ]),
  ];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-changes"]);
});

test("the lint job failing alone gives needs-lint", () => {
  const jobs = [
    job("lint", "failure", [step("Run style checks", "failure")]),
    job("mac_build_and_test", "skipped"),
  ];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-lint"]);
});

test("lint and a substantive job failing together gives needs-changes", () => {
  const jobs = [
    job("lint", "failure", [step("Run style checks", "failure")]),
    job("mac_build_and_test", "failure", [step("Build (Xcode, macOS)", "failure")]),
  ];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-changes"]);
});

test("the documentation check failing gives needs-changes", () => {
  const jobs = [
    job("lint", "success"),
    job("mac_build_and_test", "failure", [step("Verify documentation", "failure")]),
  ];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-changes"]);
});

test("the classifier's own test job failing gives needs-changes", () => {
  const jobs = [
    job("lint", "success"),
    job("ci_state_script", "failure", [step("Test the CI state classifier", "failure")]),
  ];
  const result = onComplete({ conclusion: "failure", jobs, labels: [] });
  assert.deepEqual(result.add, ["needs-changes"]);
});

test("onComplete leaves exactly one managed label and clears the rest", () => {
  const result = onComplete({
    conclusion: "failure",
    jobs: [job("lint", "failure", [step("Run style checks", "failure")])],
    labels: ["ci-running", "needs-ci", "needs-review", "approved", "cat-tools", "unread"],
  });
  assert.deepEqual(result.add, ["needs-lint"]);
  assert.deepEqual(result.remove, ["ci-running", "needs-ci", "needs-review", "approved"]);
});

const { onSynchronize } = require("./ci-state.js");

test("a scanned pull request goes back in the CI queue", () => {
  const result = onSynchronize(["needs-review", "cat-model"]);
  assert.deepEqual(result.add, ["needs-ci"]);
  assert.deepEqual(result.remove, ["needs-review"]);
});

test("a new commit clears an approval", () => {
  const result = onSynchronize(["ready-to-merge", "approved", "cat-server"]);
  assert.deepEqual(result.add, ["needs-ci"]);
  assert.deepEqual(result.remove, ["approved", "ready-to-merge"]);
});

test("a new commit clears ci-running", () => {
  const result = onSynchronize(["ci-running", "cat-misc"]);
  assert.deepEqual(result.add, ["needs-ci"]);
  assert.deepEqual(result.remove, ["ci-running"]);
});

test("a pull request already queued keeps its place and costs no write", () => {
  const result = onSynchronize(["needs-ci", "cat-model"]);
  assert.deepEqual(result.add, []);
  assert.deepEqual(result.remove, []);
});

test("an unscanned pull request is cleared but not queued", () => {
  const result = onSynchronize(["needs-review"]);
  assert.deepEqual(result.add, []);
  assert.deepEqual(result.remove, ["needs-review"]);
});

test("onSynchronize leaves labels outside its list alone", () => {
  const result = onSynchronize(["cat-model", "unread", "changes requested"]);
  assert.deepEqual(result.add, ["needs-ci"]);
  assert.deepEqual(result.remove, []);
});

const { jobSummaries, findPullRequest, applyLabels } = require("./ci-state.js");

// A fake Octokit. `paginate` reads the pages off the route function it is
// given, which is how the real client distinguishes one endpoint from another.
function makeGithub(options = {}) {
  const calls = [];
  const requests = [];
  const jobsRoute = () => {};
  jobsRoute.__pages = options.jobs ?? [];
  const pullsRoute = () => {};
  pullsRoute.__pages = options.openPulls ?? [];
  return {
    calls,
    requests,
    paginate: async (route, params) => { requests.push(params); return route.__pages ?? []; },
    rest: {
      actions: { listJobsForWorkflowRun: jobsRoute },
      pulls: { list: pullsRoute },
      repos: {
        listPullRequestsAssociatedWithCommit: async (params) => {
          requests.push(params);
          return { data: options.associated ?? [] };
        },
      },
      issues: {
        addLabels: async (params) => {
          calls.push(`add:${params.labels.join("+")}`);
        },
        removeLabel: async (params) => {
          if (options.removeError) throw options.removeError;
          calls.push(`remove:${params.name}`);
        },
      },
    },
  };
}

test("jobSummaries keeps only the fields the rules read", async () => {
  const github = makeGithub({
    jobs: [{
      id: 1,
      name: "lint",
      conclusion: "failure",
      html_url: "https://example.invalid",
      steps: [{ number: 1, name: "Run style checks", conclusion: "failure", status: "completed" }],
    }],
  });
  const jobs = await jobSummaries(github, { owner: "o", repo: "r", runId: 9 });
  assert.deepEqual(jobs, [{
    name: "lint",
    conclusion: "failure",
    steps: [{ name: "Run style checks", conclusion: "failure" }],
  }]);
  assert.deepEqual(github.requests, [{ owner: "o", repo: "r", run_id: 9, per_page: 100 }]);
});

test("jobSummaries copes with a job that reports no steps", async () => {
  const github = makeGithub({ jobs: [{ name: "lint", conclusion: "success" }] });
  const jobs = await jobSummaries(github, { owner: "o", repo: "r", runId: 9 });
  assert.deepEqual(jobs, [{ name: "lint", conclusion: "success", steps: [] }]);
});

test("findPullRequest prefers the open pull request the commit belongs to", async () => {
  const github = makeGithub({
    associated: [
      { number: 10, state: "closed", head: { sha: "aaa" }, labels: [] },
      { number: 11, state: "open", head: { sha: "aaa" }, labels: [] },
    ],
  });
  const pull = await findPullRequest(github, { owner: "o", repo: "r", headSha: "aaa" });
  assert.equal(pull.number, 11);
});

test("findPullRequest falls back to matching the head commit of an open pull request", async () => {
  const github = makeGithub({
    associated: [],
    openPulls: [
      { number: 20, head: { sha: "bbb" }, labels: [] },
      { number: 21, head: { sha: "aaa" }, labels: [] },
    ],
  });
  const pull = await findPullRequest(github, { owner: "o", repo: "r", headSha: "aaa" });
  assert.equal(pull.number, 21);
  assert.deepEqual(github.requests, [
    { owner: "o", repo: "r", commit_sha: "aaa" },
    { owner: "o", repo: "r", state: "open", per_page: 100 },
  ]);
});

test("findPullRequest prefers the association whose head is the tested commit", async () => {
  const github = makeGithub({
    associated: [
      { number: 30, state: "open", head: { sha: "ccc" }, labels: [] },
      { number: 31, state: "open", head: { sha: "aaa" }, labels: [] },
    ],
  });
  const pull = await findPullRequest(github, { owner: "o", repo: "r", headSha: "aaa" });
  assert.equal(pull.number, 31);
});

test("findPullRequest returns null when no open pull request matches", async () => {
  const github = makeGithub({ associated: [], openPulls: [] });
  const pull = await findPullRequest(github, { owner: "o", repo: "r", headSha: "aaa" });
  assert.equal(pull, null);
});

test("applyLabels adds before it removes, so a state label is never absent", async () => {
  const github = makeGithub();
  await applyLabels(github, {
    owner: "o", repo: "r", number: 5,
    add: ["needs-review"], remove: ["ci-running", "needs-ci"],
  });
  assert.deepEqual(github.calls, ["add:needs-review", "remove:ci-running", "remove:needs-ci"]);
});

test("applyLabels writes nothing when there is nothing to add", async () => {
  const github = makeGithub();
  await applyLabels(github, { owner: "o", repo: "r", number: 5, add: [], remove: [] });
  assert.deepEqual(github.calls, []);
});

test("applyLabels ignores a label that has already gone", async () => {
  const error = new Error("Label does not exist");
  error.status = 404;
  const github = makeGithub({ removeError: error });
  await applyLabels(github, {
    owner: "o", repo: "r", number: 5, add: [], remove: ["needs-ci"],
  });
  assert.deepEqual(github.calls, []);
});

test("applyLabels reports any other failure", async () => {
  const error = new Error("Forbidden");
  error.status = 403;
  const github = makeGithub({ removeError: error });
  await assert.rejects(
    applyLabels(github, { owner: "o", repo: "r", number: 5, add: [], remove: ["needs-ci"] }),
    /Forbidden/);
});
