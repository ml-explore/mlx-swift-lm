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

// The names these steps carried before they gained the prefix. A fork pull
// request runs its own copy of the CI workflow, so a copy branched before the
// rename still reports the old names. This list can go once no open pull
// request predates the rename.
const LEGACY_INFRA_STEPS = [
  "Verify MetalToolchain installed",
  "Assert Xcode 27 and the macOS 27 SDK",
  "Install MetalToolchain",
];

function isInfraStep(name) {
  return name.startsWith(INFRA_PREFIX) || LEGACY_INFRA_STEPS.includes(name);
}

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
  if (failedSteps.every(isInfraStep)) return "needs-ci";
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

async function jobSummaries(github, { owner, repo, runId }) {
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    owner, repo, run_id: runId, per_page: 100,
  });
  return jobs.map((job) => ({
    name: job.name,
    conclusion: job.conclusion,
    steps: (job.steps ?? []).map((step) => ({ name: step.name, conclusion: step.conclusion })),
  }));
}

// workflow_run.pull_requests can be empty when the head repository is a fork,
// so the number is found from the head commit instead. A commit can belong to
// more than one open pull request when branches are stacked, so prefer the one
// whose head it is: choosing a sibling makes the caller's head comparison
// discard a run that was not stale. The last resort covers a commit the
// association index has not caught up with.
async function findPullRequest(github, { owner, repo, headSha }) {
  const associated = await github.rest.repos.listPullRequestsAssociatedWithCommit({
    owner, repo, commit_sha: headSha,
  });
  const open = associated.data.find((pull) => pull.state === "open" && pull.head.sha === headSha)
    ?? associated.data.find((pull) => pull.state === "open");
  if (open) return open;

  const pulls = await github.paginate(github.rest.pulls.list, {
    owner, repo, state: "open", per_page: 100,
  });
  return pulls.find((pull) => pull.head.sha === headSha) ?? null;
}

// Adds before it removes. The reverse order leaves a window in which the pull
// request carries no state label at all, and a reader would see it as untriaged.
async function applyLabels(github, { owner, repo, number, add, remove }) {
  if (add.length > 0) {
    await github.rest.issues.addLabels({ owner, repo, issue_number: number, labels: add });
  }
  for (const name of remove) {
    try {
      await github.rest.issues.removeLabel({ owner, repo, issue_number: number, name });
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
}

module.exports = {
  STATE_LABELS, RUNNING, INFRA_PREFIX,
  onStart, onComplete, onSynchronize,
  jobSummaries, findPullRequest, applyLabels,
};
