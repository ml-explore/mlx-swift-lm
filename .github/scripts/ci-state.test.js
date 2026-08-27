"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { STATE_LABELS, RUNNING, onStart } = require("./ci-state.js");

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

const { onComplete } = require("./ci-state.js");

function step(name, conclusion) {
  return { name, conclusion };
}

function job(name, conclusion, steps = []) {
  return { name, conclusion, steps };
}

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
