// PDF viewer rules (lib/pdfView.ts): Fit to width is a real toggle, zooming leaves
// fit mode, the reading mode is remembered safely, and a scroll position maps to
// the right page. Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ACTUAL_SIZE, clampScale, effectiveScale, fitWidthScale, initialZoom, MAX_SCALE, MIN_SCALE, pageAtProbe, parseViewMode, stepScale, toggleFit, zoomBy,
} from "../lib/pdfView.ts";

// ---------- Fit to width ----------

test("the viewer starts in fit mode, which follows the measured width", () => {
  assert.equal(initialZoom.fit, true);
  assert.equal(effectiveScale(initialZoom, 1.78), 1.78);
});

test("fit mode has no scale until the page and viewer are measured", () => {
  assert.equal(effectiveScale(initialZoom, null), ACTUAL_SIZE);
});

test("clicking Fit to width again turns it off and goes back to 100%", () => {
  const off = toggleFit(initialZoom);
  assert.equal(off.fit, false);
  assert.equal(effectiveScale(off, 1.78), ACTUAL_SIZE);
  assert.equal(toggleFit(off).fit, true);
});

test("turning fit off after zooming by hand returns to that zoom, not to 100%", () => {
  let state = zoomBy(initialZoom, -1, 1.78);
  assert.equal(state.fit, false);
  state = toggleFit(toggleFit(state)); // fit on, then off again
  assert.equal(state.fit, false);
  assert.equal(effectiveScale(state, 1.78), 1.63);
});

test("zooming from fit mode continues from the scale on screen and leaves fit mode", () => {
  const out = zoomBy(initialZoom, -1, 1.78);
  assert.deepEqual(out, { fit: false, manual: 1.63 });
  const inn = zoomBy(initialZoom, 1, 1.78);
  assert.deepEqual(inn, { fit: false, manual: 1.93 });
});

test("zooming by hand keeps stepping from the manual zoom", () => {
  let state = { fit: false, manual: 1 };
  state = zoomBy(state, 1, 1.78);
  state = zoomBy(state, 1, 1.78);
  assert.equal(state.manual, 1.3);
});

test("zoom stays inside its limits", () => {
  assert.equal(stepScale(MAX_SCALE, 1), MAX_SCALE);
  assert.equal(stepScale(MIN_SCALE, -1), MIN_SCALE);
  assert.equal(clampScale(99), MAX_SCALE);
  assert.equal(clampScale(0), MIN_SCALE);
});

test("fit scale leaves a gutter and is clamped", () => {
  assert.equal(fitWidthScale(648, 600), 1);
  assert.equal(fitWidthScale(5000, 100), MAX_SCALE);
  assert.equal(fitWidthScale(60, 600), MIN_SCALE);
});

// ---------- reading mode ----------

test("only a stored 'scroll' selects continuous scrolling; anything else is one page at a time", () => {
  assert.equal(parseViewMode("scroll"), "scroll");
  assert.equal(parseViewMode("single"), "single");
  assert.equal(parseViewMode(null), "single");
  assert.equal(parseViewMode(undefined), "single");
  assert.equal(parseViewMode("garbage"), "single");
});

// ---------- which page is on screen ----------

test("a scroll position belongs to the last page that starts at or above it", () => {
  const tops = [24, 1340, 2656, 3972];
  assert.equal(pageAtProbe(tops, 0), 1);
  assert.equal(pageAtProbe(tops, 24), 1);
  assert.equal(pageAtProbe(tops, 1339), 1);
  assert.equal(pageAtProbe(tops, 1340), 2);
  assert.equal(pageAtProbe(tops, 3000), 3);
  assert.equal(pageAtProbe(tops, 99999), 4);
});

test("pages of different heights and an empty list are handled", () => {
  assert.equal(pageAtProbe([], 500), 1);
  assert.equal(pageAtProbe([0, 400, 1500], 1499), 2);
  assert.equal(pageAtProbe([0, 400, 1500], 1500), 3);
});
