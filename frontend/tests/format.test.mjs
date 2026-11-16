import test from "node:test";
import assert from "node:assert/strict";
import { clock, duration, percent, polylineSegments } from "../src/lib/format.ts";

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


test("a gap in the data breaks the plotted line instead of bridging it", () => {
  const segs = polylineSegments([0, 1, 2, 3], [1, 2, null, 4], 0, 3, 0, 4, 100, 50);
  assert.equal(segs.length, 2);
  assert.equal(segs[0].split(" ").length, 2);
  assert.equal(segs[1].split(" ").length, 1);
});
