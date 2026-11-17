set -e
: "${REPO_ROOT:?source setup_env.sh first}"
cd "$STATOR_ROOT"
S=$REPO_ROOT/script_example/03_hand_over_fork_and_agent_trees
C=$REPO_ROOT/script_example/common
D=${RUN_DATE:-$(date +%Y%m%d)}
R=$STATOR_ROOT/results
knob=/sys/kernel/mm/transparent_hugepage/shmem_enabled
old=$(sed 's/.*\[\(.*\)\].*/\1/' "$knob")
restore() { echo "$old" | sudo -n tee "$knob" >/dev/null; }
trap restore EXIT
echo advise | sudo -n tee "$knob" >/dev/null
STATOR_FATTN_HOST_GQA=2 python3 -u "$S/run_hand_over.py" --tag $D-hand-over --engine publish --repetitions 6 \
  --cells '8:80 8:320 32:320' --modes 'copy demand device_merged writable_no_read copy_map extent extent_sealed' --tokens 64
python3 "$S/check_hand_over.py" "$R/$D-hand-over"
python3 "$S/summarize_hand_over.py" "$R/$D-hand-over"
restore
trap - EXIT
STATOR_FATTN_HOST_GQA=2 python3 -u "$S/run_fault_writer.py" --tag $D-fault-writer \
  --repetitions 3 --children 8 --paragraphs 320 --tokens 96 --delay-ms 5000
python3 "$S/summarize_fault_writer.py" "$R/$D-fault-writer"
STATOR_FATTN_HOST_GQA=2 python3 -u "$S/run_live_fork.py" --tag $D-live-fork \
  --repetitions 6 --modes 'copy copy_map copy_delta extent' --paragraphs 320 --forks 8 --children 1 --step 256 \
  --tokens 64 --unlink
python3 -u "$S/run_agent_tree.py" --tag $D-agent-tree-11 --scenario tree --repetitions 6 --modes 'copy flat chain' \
  --fanouts '2,4' --context 32768 --level-paragraphs 40 --tokens 64
python3 "$C/check_scenarios.py" "$R/$D-agent-tree-11"
python3 "$S/summarize_agent_tree.py" "$R/$D-agent-tree-11" --output "$R/$D-agent-tree-11/analysis"
python3 -u "$S/run_agent_tree.py" --tag $D-agent-tree-23-31 --scenario tree --repetitions 6 --modes 'copy flat chain' \
  --fanouts '2,2,4 2,2,6' --context 32768 --level-paragraphs 20 --tokens 64
python3 "$C/check_scenarios.py" "$R/$D-agent-tree-23-31"
python3 "$S/summarize_agent_tree.py" "$R/$D-agent-tree-23-31" --output "$R/$D-agent-tree-23-31/analysis"
RESULT_TAG=$D-deep-tree bash "$S/run_deep_tree.sh"
MPS_ROOT=${TMPDIR:-/tmp}/stator-pf RESULT_TAG=$D-tree-faults bash "$S/run_tree_faults.sh"
RESULT_TAG=$D-fault-domains bash "$S/run_fault_domains.sh"
SERVERS="2 4 8" RESULT_TAG=$D-batch-servers bash "$S/run_batch_servers.sh"
RESULT_TAG=$D-vmm-attach bash "$S/run_vmm_attach.sh"
