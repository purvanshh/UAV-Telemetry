import test from "node:test";
import assert from "node:assert/strict";
import { clock, duration, percent, verdictLabel, verdictTone, polylineSegments } from "../src/lib/format.ts";

test("clock formats mm:ss and h:mm:ss", () => {
  assert.equal(clock(0), "00:00");
  assert.equal(clock(92.4), "01:32");
  assert.equal(clock(3725), "1:02:05");
  assert.equal(clock(null), "—");
  assert.equal(clock(Number.NaN), "—");
});

test("duration and percent handle missing values", () => {
  assert.equal(duration(5.9), "5.9 s");
  assert.equal(duration(125), "2 min 5 s");
  assert.equal(duration(undefined), "—");
  assert.equal(percent(0.456), "46%");
  assert.equal(percent(null), "—");
});

test("verdict labels never claim normal or safe", () => {
  const states = [
    "anomalies_detected", "no_alerts_adequate_coverage", "no_alerts_low_coverage",
    "analysis_failed", "insufficient_data", "not_analyzed", null,
  ];
  for (const s of states) {
    const label = verdictLabel(s).toLowerCase();
    assert.ok(!label.includes("normal") && !label.includes("safe"), label);
  }
  assert.equal(verdictTone("no_alerts_low_coverage"), "caution");
  assert.equal(verdictTone("anomalies_detected"), "attention");
  assert.equal(verdictTone("analysis_failed"), "error");
});

test("a gap in the data breaks the plotted line instead of bridging it", () => {
  const segs = polylineSegments([0, 1, 2, 3], [1, 2, null, 4], 0, 3, 0, 4, 100, 50);
  assert.equal(segs.length, 2);
  assert.equal(segs[0].split(" ").length, 2);
  assert.equal(segs[1].split(" ").length, 1);
});
