import type { AgentProgress } from "@/lib/api";

/** What each telemetry topic is, in words a pilot uses. */
const TOPIC: Record<string, string> = {
  arjuna_vibration: "vibration",
  arjuna_imu: "motion-sensor",
  arjuna_attitude: "attitude",
  arjuna_actuator_outputs: "motor",
  arjuna_battery: "battery",
  arjuna_gps: "GPS",
  arjuna_position: "height and position",
  arjuna_baro: "barometer",
  arjuna_estimator_innov: "position-estimate",
  arjuna_rc_input: "remote-control",
  arjuna_controller: "control-loop",
  arjuna_vehicle_status: "flight-mode",
  arjuna_setpoints: "command",
  arjuna_mag: "compass",
  arjuna_rangefinder: "height-sensor",
};

const AXIS: Record<string, string> = { x: "horizontal position", y: "horizontal position", z: "height" };
const axis = (a: Record<string, unknown>) => AXIS[String(a.axis)] ?? String(a.axis ?? "attitude");

function topic(args: Record<string, unknown>, key = "topic"): string {
  return TOPIC[String(args[key] ?? "")] ?? "flight";
}

/**
 * The sentence for the step the review is really on. Only states what the agent is doing at this moment,
 * taken from the server, never a guess based on time.
 */
export function stepMessage(p: AgentProgress | null): string {
  if (!p || !p.running) return "Starting the detailed review…";
  if (p.step === "starting") return "Starting the detailed review…";
  if (p.step === "thinking") {
    if (!p.calls) return "Planning the review of this UAV flight…";
    const what = p.tool ? looked(p.tool, p.args ?? {}) : null;
    // the model is reading the result of its last check and choosing the next one
    return what ? `Weighing what ${what} showed, and choosing the next check…` : "Deciding what to check next…";
  }
  if (p.step === "validating") return "Fact-checking every number in the explanation…";
  const a = p.args ?? {};
  switch (p.tool) {
    case "get_flight_overview":
      return "Reading the flight overview: modes, take-off and landing…";
    case "get_findings":
      return "Going through the problems the automatic checks found…";
    case "get_physics":
      return "Checking motor balance and thrust against the drone's physics…";
    case "match_pilot_report":
      return "Matching what you noticed to the flight timeline…";
    case "list_signals":
      return "Looking up which telemetry this drone recorded…";
    case "signal_stats":
      return `Examining the ${topic(a)} telemetry…`;
    case "compare_periods":
      return `Comparing ${topic(a)} telemetry before and during the problem…`;
    case "find_periods":
      return `Searching the flight for unusual ${topic(a)} readings…`;
    case "dominant_frequency":
      return `Looking for oscillation in the ${topic(a)} telemetry…`;
    case "correlate":
      return `Checking how ${topic(a, "a_topic")} and ${topic(a, "b_topic")} readings relate…`;
    case "controller_tracking":
      return `Checking how well the drone held its ${axis(a)}…`;
    case "motor_outputs":
      return "Checking the output of each motor…";
    case "get_messages":
      return "Reading the flight controller's messages…";
    case "sequence_of_departures":
      return "Working out which reading changed first…";
    case "state_at":
      return "Reconstructing the drone's state at key moments…";
    default:
      return "Analysing the flight telemetry…";
  }
}

/** What the last check looked at, as a noun phrase. */
function looked(tool: string, a: Record<string, unknown>): string | null {
  switch (tool) {
    case "get_flight_overview":
      return "the flight overview";
    case "get_findings":
      return "the automatic checks";
    case "get_physics":
      return "the motor-balance check";
    case "match_pilot_report":
      return "your report";
    case "signal_stats":
    case "compare_periods":
    case "find_periods":
    case "dominant_frequency":
      return `the ${topic(a)} telemetry`;
    case "correlate":
      return `the ${topic(a, "a_topic")} and ${topic(a, "b_topic")} telemetry`;
    case "controller_tracking":
      return `the ${axis(a)} control`;
    case "motor_outputs":
      return "the motor outputs";
    case "get_messages":
      return "the flight controller's messages";
    case "sequence_of_departures":
      return "the order things went wrong";
    case "state_at":
      return "the drone's state at that moment";
    default:
      return null;
  }
}
