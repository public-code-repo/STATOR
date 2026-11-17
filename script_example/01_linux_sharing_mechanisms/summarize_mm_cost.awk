function field(name, i, pair) {
  for (i = 1; i <= NF; ++i) {
    split($i, pair, "=")
    if (pair[1] == name) return pair[2]
  }
  return ""
}

function t95(samples, df) {
  df = samples - 1
  if (df <= 1) return 12.706
  if (df == 2) return 4.303
  if (df == 3) return 3.182
  if (df == 4) return 2.776
  if (df == 5) return 2.571
  if (df == 6) return 2.447
  if (df == 7) return 2.365
  return 2.262
}

/^BEGIN_MM_COST / { run = field("run"); next }
/^PROBE_FAILED / { failed[field("operation") "," 0 + field("threads")]++; next }

/^probe=mm_cost / {
  key = field("operation") "," field("threads")
  phase = field("phase")
  per_gib = field("ms") * 1073741824 / field("bytes")
  value[key SUBSEP phase SUBSEP run] = per_gib
  total[key SUBSEP phase] += per_gib
  count[key SUBSEP phase]++
  per_event[key SUBSEP phase] += field("us_per_event")
  events[key] = field("events") * 1073741824 / field("bytes")
  seen[key] = 1
  if (run > runs) runs = run
}

END {
  print "operation,threads,runs,events_per_gib,no_cuda_ms_per_gib," \
        "cuda_ms_per_gib,no_cuda_us_per_event,cuda_us_per_event," \
        "extra_us_per_event,ratio,lower_ci95,upper_ci95"
  fflush()
  sorter = "LC_ALL=C sort -t, -k1,1 -k2,2n"
  for (key in seen) {
    samples = 0
    sum = 0
    sumsq = 0
    for (r = 1; r <= runs; ++r) {
      a = value[key SUBSEP "cuda" SUBSEP r]
      b = value[key SUBSEP "no_cuda" SUBSEP r]
      if (a <= 0 || b <= 0) continue
      samples++
      sum += log(a / b)
      sumsq += log(a / b) * log(a / b)
    }
    if (samples == 0) continue
    average = sum / samples
    variance = 0
    if (samples > 1) variance = (sumsq - sum * sum / samples) / (samples - 1)
    if (variance < 0) variance = 0
    margin = 0
    if (samples > 1) margin = t95(samples) * sqrt(variance / samples)
    before = total[key SUBSEP "no_cuda"] / count[key SUBSEP "no_cuda"]
    after = total[key SUBSEP "cuda"] / count[key SUBSEP "cuda"]
    before_event = per_event[key SUBSEP "no_cuda"] / count[key SUBSEP "no_cuda"]
    after_event = per_event[key SUBSEP "cuda"] / count[key SUBSEP "cuda"]
    printf "%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", key, samples, \
           events[key], before, after, before_event, after_event, \
           after_event - before_event, exp(average), exp(average - margin), \
           exp(average + margin) | sorter
  }
  close(sorter)
}
