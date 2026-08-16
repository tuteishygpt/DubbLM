# Start Dubbing Job Design

## Goal

Replace the Workflow action label `Queue job` with `Start` and make its intent
clear: one click submits the media and every currently selected workflow option
to the existing dubbing-job queue.

## Design

`WorkflowView` remains the sole UI owner of submission. Its form submit handler
will keep uploading the selected video and isolated tracks, construct overrides
from the current form state (including enabled voice profiles), and create the
job through `POST /api/jobs`. The existing backend queue begins the selected
pipeline mode from the persisted job configuration.

The action button will be renamed to `Start`. While submission is in progress,
the button will be disabled and show `Starting…`; this prevents duplicate job
creation while uploads or the job request are pending. On success and failure,
the control returns to its normal state and the existing status/error message is
shown.

## Boundaries

- No new backend endpoint or queue implementation is needed.
- The selected run mode and every visible workflow option retain their existing
  semantics and are sent with the created job.
- Validation errors continue to prevent submission before uploads begin.

## Tests

Add focused UI coverage for the renamed Start button, preservation of selected
workflow overrides, and duplicate-submission prevention while an upload is
pending. Run the focused Vitest suite, TypeScript typecheck, and frontend build.
