"use strict";

// The script removes only a label on this list. It does not touch any other
// label.
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

// approved and ready-to-merge are absent on purpose. onComplete needs to see an
// approval that arrived during the run.
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

// Never ask GitHub to add a label the pull request already has.
function missing(labels, candidates) {
  const held = new Set(labels);
  return candidates.filter((name) => !held.has(name));
}

function onStart(labels) {
  return { add: missing(labels, [RUNNING]), remove: present(labels, PREVIOUS_OUTCOME) };
}

// A step whose name starts with this prefix reports a problem in the runner
// image. It does not report a problem in the contribution. Only a re-run can
// clear it, so it must never ask the author for a change.
const INFRA_PREFIX = "Infra: ";

// These are the old step names, from before the prefix. A fork pull request runs
// its own copy of the CI workflow, so a copy branched before the rename still
// reports the old names. Delete this list when no open pull request is older
// than the rename.
const LEGACY_INFRA_STEPS = [
  "Verify MetalToolchain installed",
  "Assert Xcode 27 and the macOS 27 SDK",
  "Install MetalToolchain",
];

function isInfraStep(name) {
  return name.startsWith(INFRA_PREFIX) || LEGACY_INFRA_STEPS.includes(name);
}

// Do not reorder these tests. A failed job with no failed step means a timeout
// or a lost runner. That result asks for another run, and never asks the author
// for a change. The build-machine test comes before the lint test, so a
// MetalToolchain failure never counts as a formatting problem. needs-lint says
// formatting is the only thing to fix, so only lint failing alone gets it.
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

// Only a triaged pull request goes back in the queue, and a cat-* label is the
// mark of that. If no cat-* label is present, the stale labels still go, and the
// script adds nothing.
function onSynchronize(labels) {
  const scanned = labels.some((name) => name.startsWith("cat-"));
  // Two lists on purpose. `keep` says what not to remove. The add list says what
  // to write, and is empty when the label is already there. If you merge the two
  // lists, a pull request that already has needs-ci loses it.
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

// GitHub can send an empty pull request list for a fork, so the number comes
// from the head commit instead. One commit can belong to two open pull requests
// when branches are stacked. If this picks the wrong one, its head has moved,
// the caller treats the run as stale, and neither pull request gets a label. The
// last search covers a commit GitHub has not linked yet.
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

// This function adds labels before it removes labels. In the other order, the
// pull request has no label for a short time. A reader of the list then sees it
// as untriaged.
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
