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

// A step whose name starts with this prefix reports the runner image, not the
// contribution. Only a re-run can clear it, so it must never ask the author
// for a change.
const INFRA_PREFIX = "Infra: ";

// The three orderings below carry the whole correctness of this function.
// The no-failed-step test comes first, so a timeout or a lost runner reads as
// "run it again" rather than as a fault. The infrastructure test comes before
// the lint test, so a MetalToolchain failure never reads as bad formatting.
// needs-lint claims formatting is the only thing to fix, so it requires lint to
// be the only failed job. Any other failure alongside it means more work than
// that label admits.
function classify({ conclusion, jobs, labels }) {
  const failedJobs = jobs.filter((job) => job.conclusion === "failure");

  if (failedJobs.length === 0) {
    if (conclusion !== "success") return "needs-ci";
    const approved = labels.includes("approved") || labels.includes("ready-to-merge");
    return approved ? "ready-to-merge" : "needs-review";
  }

  const failedSteps = failedJobs.flatMap((job) =>
    (job.steps ?? [])
      .filter((step) => step.conclusion === "failure")
      .map((step) => step.name));

  if (failedSteps.length === 0) return "needs-ci";
  if (failedSteps.every((name) => name.startsWith(INFRA_PREFIX))) return "needs-ci";
  if (failedJobs.length === 1 && failedJobs[0].name === "lint") return "needs-lint";
  return "needs-changes";
}

function onComplete({ conclusion, jobs, labels }) {
  const chosen = classify({ conclusion, jobs, labels });
  const candidates = [RUNNING, ...STATE_LABELS.filter((name) => name !== chosen)];
  return { add: missing(labels, [chosen]), remove: present(labels, candidates) };
}

// A cat-* label proves a scan finished. Without this guard, a contributor who
// pushes twice before the scanner reaches them would be queued for CI on code
// no scan has read, and CI runs contributor code on the self-hosted mac.
function onSynchronize(labels) {
  const scanned = labels.some((name) => name.startsWith("cat-"));
  // `keep` decides what survives the clear, and the add list is derived from it
  // separately. Driving both from one list would delete the needs-ci this means
  // to keep, as soon as the add was suppressed for already being there.
  const keep = scanned ? ["needs-ci"] : [];
  const candidates = [RUNNING, ...STATE_LABELS.filter((name) => !keep.includes(name))];
  return { add: missing(labels, keep), remove: present(labels, candidates) };
}

module.exports = {
  STATE_LABELS, RUNNING, INFRA_PREFIX,
  onStart, onComplete, onSynchronize,
};
