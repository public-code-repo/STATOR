function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == name) return pair[2] }
  return ""
}
/^BEGIN_KVBATCHFAULT / { run = field("run") + 0; if (run > runs) runs = run; next }
/^CASE / {
  config = field("config"); fault = field("fault"); key = config SUBSEP fault
  cases[key]++; trigger[key] += field("trigger_ms"); case_failed = 0; case_tps = 0
  keys[key] = 1
  next
}
/^FAULT / { if (field("status") == "cudaErrorIllegalAddress") illegal[key]++; next }
/^SERVER / {
  server = field("name")
  if (field("exit") != 0 || field("generation_tps") == "") { failed[key]++; case_failed = 1 }
  else case_tps += field("generation_tps")
  next
}
/^AGENT / {
  texts[key, server]++
  text[config, fault, run, field("index")] = field("text")
  if (server == "b") b_agents[config, fault, run, field("index")] = 1
  next
}
/^END_CASE / { if (!case_failed) { ok[key]++; tps[key] += case_tps } next }
END {
  print "config,fault,cases,illegal_address_cases,servers_failed,texts_server_a,texts_server_b,b_texts_equal_to_none,b_texts_compared,trigger_ms_mean,generation_tps_total"
  for (key in keys) {
    split(key, part, SUBSEP)
    equal = 0; compared = 0
    for (k in b_agents) {
      split(k, q, SUBSEP)
      if (q[1] != part[1] || q[2] != part[2]) continue
      if ((q[1], "none", q[3], q[4]) in text) {
        compared++
        if (text[q[1], q[2], q[3], q[4]] == text[q[1], "none", q[3], q[4]]) equal++
      }
    }
    printf "%s,%s,%d,%d,%d,%d,%d,%d,%d,%.0f,%s\n", part[1], part[2], cases[key],
      illegal[key] + 0, failed[key] + 0, texts[key, "a"] + 0, texts[key, "b"] + 0,
      equal, compared, trigger[key] / cases[key],
      (ok[key] ? sprintf("%.2f", tps[key] / ok[key]) : "NA")
  }
}
