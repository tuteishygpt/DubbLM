# Dark Minimal React UI Design

## Goal

Turn the DubbLM React SPA into a modern dark, minimal interface while keeping every existing workflow, API contract, and accessible control intact. The design should make long-running dubbing work calm to scan: the workflow is clear, the queue is easy to monitor, and dense editing views remain usable.

## Scope

The work applies to the React client in `frontend/`. It covers the application shell, navigation, forms, controls, status feedback, tables, and responsive behavior for Workflow, Jobs, Settings, Voice Profiles, and Dubbing Texts.

It does not change FastAPI routes, client request shapes, persisted data, job behavior, or any business logic. It also does not add a component library, a state library, icons package, custom font, animation framework, or image assets.

## Visual direction

Use a restrained "control room" visual language:

- near-black page canvas, layered graphite surfaces, and subtle borders;
- one cool blue accent for the primary action and current navigation item;
- semantic colors reserved for queued/running/succeeded/failed state and destructive actions;
- neutral system typography, generous vertical rhythm, and minimal shadows;
- no gradients, decorative illustrations, oversized headings, or excessive cards.

Define the palette, spacing, radii, and shadow values as CSS custom properties in `frontend/src/styles.css`. Component styles consume those tokens instead of repeating color literals.

## Application shell and navigation

`App` renders a stable shell with a branded top bar, the current view title, and a compact local status indicator. The primary navigation becomes a vertical sidebar on wide layouts; each item remains a native button with a text label and the existing `aria-current="page"` state. On narrow viewports it becomes a horizontally scrollable top navigation so controls stay reachable without a hidden menu.

The main content area is centered but no longer artificially narrow for data-heavy screens. Its maximum width is appropriate for tables and uses responsive padding. Loading, error, and status messages use visibly distinct but quiet inline surfaces and retain their current ARIA roles.

## Shared components and interaction rules

Use native inputs, selects, buttons, fieldsets, and tables. Add presentational class names only where needed; do not alter labels, accessible names, button text, route calls, or event behavior.

- Buttons have primary, secondary, and danger variants; disabled controls remain clearly legible.
- Inputs use a dark filled surface, clear focus outline, and concise labels. Related settings use grouped fieldsets with legends.
- Status values render as semantic chips: queued, running, succeeded, failed, and neutral.
- Surface panels use modest padding and a thin border. They group meaningful units, not every individual field.
- Tables use a sticky header where scrolling is possible, muted grid lines, consistent numeric alignment, and a scroll container on constrained screens.

## View treatment

### Workflow

Present upload and job setup as the primary panel. Put media selection and speaker mapping first, then workflow options in a responsive grid, and finish with the single `Queue job` action. Less frequent workflow settings remain visible and functional but sit in a clearly labelled secondary settings group.

### Jobs

Use state chips in the job list and compact file links. Keep live connection state near the title. Render logs in a dark monospace scroll panel with stable height so new events do not distort the page.

### Settings

Group schema fields by their existing sections or scope when available; otherwise preserve schema order in a responsive form grid. The save action stays easy to find at the end of the form and on long screens may also be repeated in the section header without changing its request behavior.

### Voice Profiles

Separate profile editing, reference upload/library, and assignment into three panels. Preserve the existing profile selection, dependent provider/model/voice options, upload, delete, and assignment behavior.

### Dubbing Texts

Keep the editor table as the center of the screen. Place job selection, source information, dirty state, save, and regeneration controls in a compact toolbar above it. Preserve editable cells, audio playback links, selection radio buttons, and the read-only audio column.

## Responsive and accessibility behavior

At desktop widths, forms may use two columns and the sidebar is visible. At tablet and phone widths, grids collapse to one column, tables scroll within their own container, and the navigation switches to its compact horizontal form. Keyboard focus has a high-contrast visible ring, color is never the only source of a status meaning, and all current labels and ARIA status/error semantics remain unchanged.

## Verification

Retain and run the existing frontend unit tests and TypeScript check. Add or update focused tests only when markup changes affect an accessibility contract or a user-visible semantic element. Build the SPA after styling changes to catch CSS/import regressions. Manually review Workflow, Jobs, Settings, Voice Profiles, and Dubbing Texts at desktop and narrow viewport widths.
