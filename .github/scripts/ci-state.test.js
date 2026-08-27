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
