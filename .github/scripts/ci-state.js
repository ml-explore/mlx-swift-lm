"use strict";

// The labels this script manages. It never removes a label outside this list,
// so a separate effort on another label cannot collide with it.
const STATE_LABELS = [
  "needs-scan",
  "needs-ci",
  "needs-lint",
  "needs-changes",
  "needs-review",
  "approved",
  "ready-to-merge",
];

const RUNNING = "ci-running";

// The labels that describe the run that just ended. approved and
// ready-to-merge survive a start on purpose: a maintainer may approve while a
// run is in progress, and onComplete reads that as evidence.
const PREVIOUS_OUTCOME = [
  "needs-ci",
  "needs-lint",
  "needs-changes",
  "needs-review",
];

function present(labels, candidates) {
  const held = new Set(labels);
  return candidates.filter((name) => held.has(name));
}

// Never ask GitHub to add a label the pull request already holds. The add is a
// no-op there, so suppressing it means a decision that changes nothing costs no
// API call at all.
function missing(labels, candidates) {
  const held = new Set(labels);
  return candidates.filter((name) => !held.has(name));
}

function onStart(labels) {
  return { add: missing(labels, [RUNNING]), remove: present(labels, PREVIOUS_OUTCOME) };
}

module.exports = { STATE_LABELS, RUNNING, onStart };
